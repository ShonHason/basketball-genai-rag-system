old Rerank function : 
@retry(wait=wait)
def rerank(question, chunks):
    """Reranks retrieved chunks using the LLM to prioritize the most relevant information."""
    if not chunks: return []
    system_prompt = "You are a sports document re-ranker. Rank chunks by relevance. Reply only with JSON."
    user_prompt = f"Question: {question}\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    
    response = completion(model=MODEL, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], response_format=RankOrder)
    
    try:
        order = RankOrder.model_validate_json(response.choices[0].message.content).order
        return [chunks[i - 1] for i in order if i <= len(chunks)]
    except Exception as e:
        print(f"[DEBUG] Rerank parsing failed: {e}")
        return chunks

old fetch_context :

def fetch_context(original_question, clean_history=[]):
    """Advanced retrieval: Query Rewriting -> Vector Search -> Merging -> Reranking."""
    # Use history-aware rewriting for better context memory
    rewritten_question = rewrite_query(original_question, clean_history)
    
    # Perform dual search (original and rewritten)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    
    # Merge and deduplicate chunks based on content
    merged = chunks1[:]
    seen_content = {c.page_content for c in chunks1}
    for chunk in chunks2:
        if chunk.page_content not in seen_content:
            merged.append(chunk)
            seen_content.add(chunk.page_content)
            
    # Apply reranking to find the top FINAL_K chunks
    return rerank(original_question, merged)[:FINAL_K]






Run -> Retrival 40, K=5, AVERAGE_CHUNK_SIZE = 800 , Overload = 100
4500 seconds

Mean Reciprocal Rank (MRR)
0.8290
Normalized DCG (nDCG)
0.8384
Keyword Coverage
89.5%
Accuracy
4.03/5
Completeness
3.87/5
Relevance
3.97/5

Run -> Retrival 20, K=5, AVERAGE_CHUNK_SIZE = 800 , Overload = 100
Mean Reciprocal Rank (MRR)
0.8104
Normalized DCG (nDCG)
0.8199
Keyword Coverage
87.9%
Accuracy
3.85/5
Completeness
3.77/5
Relevance
3.79/5

Run -> Retrival 40, K=5, AVERAGE_CHUNK_SIZE = 800 , Overload = 100  changed version MODEL = "openai/gpt-4o-mini": 
Mean Reciprocal Rank (MRR)
0.8300
Normalized DCG (nDCG)
0.8376
Keyword Coverage
91.9%
Accuracy
3.94/5
Completeness
3.97/5
Relevance
4.19/5


@retry(wait=wait)
def rerank(question, chunks):
    """
    Advanced Reranker: Acts as an aggressive filter to remove noise from K=50 retrieval.
    Focuses on Entity matching, Data Density (tables), and Source relevance.
    """
    if not chunks: 
        return []
    
    # System prompt designed to force the LLM to prioritize hard facts over summaries
    system_prompt = """
    You are a High-Precision Sports Data Analyst for the Hoopball League.
    Your task is to identify which text chunks contain the EXACT facts needed to answer the question.

    STRICT RANKING RULES:
    1. DIRECT ENTITY MATCH: Chunks containing the specific player or team name from the question MUST be ranked #1-#3.
    2. DATA DENSITY: Prioritize chunks with Markdown tables over text summaries.
    3. COMPARISON DIVERSITY: If comparing two players, ensure chunks for BOTH players are in the top 5.
    4. SOURCE PRIORITY: Favor chunks from specific player files (e.g., 'player_name.md') over general league stats.

    Return ONLY the RankOrder JSON: {"order": [ID1, ID2, ...]}
    """

    user_prompt = f"Question: {question}\n\n"
    for i, chunk in enumerate(chunks):
        # Extracting the filename from metadata helps the LLM recognize player-specific files
        source = chunk.metadata.get('source', 'unknown')
        user_prompt += f"### CHUNK ID {i+1} [Source: {source}]:\n{chunk.page_content}\n\n"
    
    try:
        response = completion(
            model=MODEL, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=RankOrder,
            # temperature=0 is critical for consistency and prevents "hallucinated" rankings
            temperature=0 
        )
        
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order
        
        # Build the ranked list using the original Result objects
        reranked = [chunks[i - 1] for i in order if 0 < i <= len(chunks)]
        
        # Fallback: If the LLM didn't rank all chunks, append the missing ones at the end
        # This ensures we don't lose potential evidence if the LLM was lazy
        seen_indices = {i-1 for i in order if 0 < i <= len(chunks)}
        for i in range(len(chunks)):
            if i not in seen_indices:
                reranked.append(chunks[i])
                
        return reranked

    except Exception as e:
        print(f"[DEBUG] Rerank parsing failed: {e}")
        return chunks


Run -> Retrival 40, K=5, AVERAGE_CHUNK_SIZE = 800 , Overload = 100  changed version MODEL = "openai/gpt-4o-mini"+changed rerank function: 
3000sec
Mean Reciprocal Rank (MRR)
0.8432
Normalized DCG (nDCG)
0.8549
Keyword Coverage
93.8%
Accuracy
3.98/5
Completeness
3.97/5
Relevance
4.27/5
Run -> Retrival 45, K=8, AVERAGE_CHUNK_SIZE = 800 , Overload = 100  changed version MODEL = "openai/gpt-4o-mini"+changed rerank function: 
Mean Reciprocal Rank (MRR)
0.8233
Normalized DCG (nDCG)
0.8294
Keyword Coverage
94.4%
Accuracy
3.98/5
Completeness
3.94/5
Relevance
4.29/5
Run -> Retrival 45, K=5, AVERAGE_CHUNK_SIZE = 800 , Overload = 100 , MODEL = gpt-4o-mini 
 +changed rerank function: 
Mean Reciprocal Rank (MRR)
0.8265
Normalized DCG (nDCG)
0.8371
Keyword Coverage
92.3%
Accuracy
3.89/5
Completeness
3.89/5
Relevance
4.29/5
gpt-oss-120b
Run -> Retrival 20, K=5, AVERAGE_CHUNK_SIZE = 800 , Overload = 100 ,MODEL = gpt-4o-mini 
Mean Reciprocal Rank (MRR)
0.8254
Normalized DCG (nDCG)
0.8371
Keyword Coverage
92.7%   


Run -> Retrival 20, K=7, AVERAGE_CHUNK_SIZE = 800 , Overload = 100 ,MODEL = gpt-4o-mini 
Mean Reciprocal Rank (MRR)
0.8306
Normalized DCG (nDCG)
0.8404
Keyword Coverage
94.4%

Run -> Retrival 20, K=7, AVERAGE_CHUNK_SIZE = 800 , Overload = 100 ,MODEL = gpt-4o-mini (change code)
Mean Reciprocal Rank (MRR)
0.8298
Normalized DCG (nDCG)
0.8388
Keyword Coverage
94.2%


Run -> Retrival 15, K=7, AVERAGE_CHUNK_SIZE = 800 , Overload = 100 ,MODEL = gpt-4o-mini 
Mean Reciprocal Rank (MRR)
0.8256
Normalized DCG (nDCG)
0.8405
Keyword Coverage
94.3%