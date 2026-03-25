import random

FIRST_NAMES = {
    "USA": [
        "Dante","Marcus","Tyrese","Jalen","Malik","Andre","Kendall","Jordan","Isaiah","Miles",
        "Devin","Caleb","Brandon","Darius","Terrell","Zachary","Austin","Derrick","Jason","Trevor"
    ],
    "Spain": [
        "Alejandro","Sergio","Pablo","Diego","Carlos","Miguel","Luis","Javier",
        "Fernando","Raul","Manuel","Adrian","Daniel","Ruben","Alberto","Victor"
    ],
    "France": [
        "Lucas","Theo","Nicolas","Antoine","Mathieu","Julien","Hugo","Louis",
        "Pierre","Maxime","Bastien","Florian","Quentin","Adrien","Romain"
    ],
    "Serbia": [
        "Nikola","Bogdan","Milos","Stefan","Marko","Nemanja","Luka","Filip",
        "Vladimir","Milan","Aleksandar","Petar","Dusan"
    ],
    "Italy": [
        "Luca","Marco","Giovanni","Alessandro","Matteo","Davide","Francesco",
        "Andrea","Stefano","Simone","Paolo","Riccardo","Giorgio"
    ],
    "Germany": [
        "Lukas","Jonas","Felix","Leon","Tim","Maximilian","David","Julian",
        "Sebastian","Niklas","Moritz","Tobias","Alexander"
    ],
    "Japan": [
        "Hiroshi","Takumi","Daichi","Kenta","Riku","Yuki","Sota","Haruto",
        "Ren","Kaito","Takeru","Shota"
    ],
    "Mexico": [
        "Carlos","Javier","Diego","Fernando","Luis","Andres","Eduardo",
        "Roberto","Angel","Ricardo","Martin","Hector"
    ],
    "Brazil": [
        "Joao","Lucas","Rafael","Mateus","Gabriel","Bruno","Pedro","Thiago",
        "Felipe","Andre","Rodrigo","Caio"
    ],
    "Canada": [
        "Ethan","Logan","Noah","Liam","Connor","Ryan","Nathan","Dylan",
        "Cole","Tyler","Brandon","Cameron"
    ],
}


LAST_NAMES = {
    "USA": [
        "Rivers","Hayes","Cole","Parker","Brooks","Walker","Mitchell","Turner",
        "Harris","Reed","Powell","Bennett","Ward","Bryant","Foster","Griffin",
        "Ellis","Howard","Russell","Morris"
    ],
    "Spain": [
        "Garcia","Fernandez","Lopez","Martinez","Sanchez","Perez","Gomez",
        "Diaz","Torres","Ramirez","Navarro","Castro","Ortega"
    ],
    "France": [
        "Dubois","Moreau","Laurent","Petit","Roux","Girard","Andre","Mercier",
        "Bonnet","Lambert","Fontaine","Chevalier"
    ],
    "Serbia": [
        "Jovic","Bogdanovic","Petrovic","Nikolic","Markovic","Milosevic",
        "Stojanovic","Pavlovic","Radovanovic"
    ],
    "Italy": [
        "Moretti","Ricci","Romano","Ferrari","Greco","Marino","Bruno",
        "Conti","Galli","Costa","Mancini"
    ],
    "Germany": [
        "Wagner","Schmidt","Fischer","Becker","Hoffmann","Koch",
        "Bauer","Klein","Wolf","Neumann"
    ],
    "Japan": [
        "Tanaka","Suzuki","Yamamoto","Kobayashi","Sato","Ito",
        "Watanabe","Nakamura","Kato","Yoshida"
    ],
    "Mexico": [
        "Mendoza","Torres","Ramirez","Vargas","Castillo","Morales",
        "Gutierrez","Ortiz","Santos"
    ],
    "Brazil": [
        "Silva","Oliveira","Costa","Ribeiro","Carvalho","Almeida",
        "Teixeira","Barbosa","Freitas"
    ],
    "Canada": [
        "Anderson","Campbell","Thompson","Bennett","Walker",
        "Morgan","Cooper","Richardson","Sullivan"
    ],
}

def generate_name_by_country(country):

    first_pool = FIRST_NAMES.get(country, FIRST_NAMES["USA"])
    last_pool = LAST_NAMES.get(country, LAST_NAMES["USA"])

    first = random.choice(first_pool)
    last = random.choice(last_pool)

    return f"{first} {last}"