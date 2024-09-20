

#1. Läsa in maträtter från fil och spara det i en lista

def läsa_in_maträtter_från_fil(filnamn):


    #Öppna vår maträtt fil
    maträtt_fil = open(filnamn, "r")

    #Definera listan
    maträtt_lista = []


    #Går igenom alla rader i maträtt filen
    while True:
        #Läs in maträtten som finns i varje rad
        maträtt = maträtt_fil.readline().strip("\n")

        #Om vi hittar en rad som är tom så innebär det att filen är slut
        if maträtt == "":
            break
        print("väntar på att skriva")

        #Sparar varje maträtt till vår lista
        maträtt_lista.append(maträtt)



    #Stäng filen
    maträtt_fil.close()


    return maträtt_lista



#2. Kunna lägga till maträtter till lista
def lägg_till_maträtter_till_lista(maträtt_lista, maträtt_att_läggas_till):
    maträtt_lista.append(maträtt_att_läggas_till)

    return maträtt_lista



#3. Spara vår lista till fil
def spara_lista_till_fil(maträtt_lista, filnamn):

    #Öppna filen så vi kan skriva till den
    maträtt_fil = open(filnamn, "w")


    #Iterera genom vår lista
    for i in range(len(maträtt_lista)):

        #För varje element i vår maträttlista så skriver vi det till filen
        #\n lägger man till så att varje maträtt skrivs på ny rad
        maträtt_fil.write(maträtt_lista[i] + "\n")


    #Stäng filen
    maträtt_fil.close()




def huvudprogram():

    filnamn = "mat_fil.txt"

    #Läsa in från fil
    maträtt_lista = läsa_in_maträtter_från_fil(filnamn)


    #Spara till lista
    maträtt_att_läggas_till = input("Ange en maträtt som ska sparas")
    maträtt_lista = lägg_till_maträtter_till_lista(maträtt_lista, maträtt_att_läggas_till)

    #Spara lista till fil
    spara_lista_till_fil(maträtt_lista, filnamn)

    print("hello")



huvudprogram()
print("saeg")



