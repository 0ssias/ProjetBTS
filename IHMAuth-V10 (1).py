from tkinter import * # Pour l'IHM
import tkinter.messagebox # Pour l'IHM
import serial # Pour le capteur
import time # Pour l'acquisition
import MySQLdb as mc # Pour la connexion a la bdd
import ClientServiceWeb # Pour la connexion au service web client & serveur
import os # Pour connaitre le chemin d'accès du fichier config
import signaturecomparator # Pour la comparaison des signatures cardiaques
import socket # Pour accéder à l'addresse locale
import paho.mqtt.client as mqtt # Pour l'envoie des données au Broker

class IHM(Tk):
    def getlocalip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # socket.AF_INET est pour IpV4 et socket.SOCK_DGRAM pour UDP.
        try:
            s.connect(('10.255.255.255', 1)) # Fausse adresse servant simplement a savoir l'adresse ip locale du poste. 
            ip = s.getsockname()[0] # Donne l'addresse IP associer au socket (donc l'adresse locale.)
        except Exception: # Erreur si 127.0.0.1
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

    def sendconnectioninfo(self,disconnect):
        broker_address = "10.78.5.111" # Adresse IP du Broker
        port = 1883 # Port IP du Broker
        topic = "employé/connexion" # Topic
        if disconnect == 'logout': # Envoie une connexion si deconnect est égale a False
            message = "Connexion"
        elif disconnect == "login": # Envoie une connexion si deconnect est égale a True
            message = "Deconnexion"
        client = mqtt.Client() # Créer le client MQTT
        try:
            # Connexion au broker
            client.connect(broker_address, port, 60)

            # Publier le message
            result = client.publish(topic, message)
            status = result[0]

            if status == 0:
                print(f"Message publié avec succès sur le topic '{topic}'")
            else:
                print(f"Échec de publication du message sur le topic '{topic}'")
                self.error(10) # Envoie un message d'erreur en cas d'erreur

            client.disconnect()

        except Exception as e:
            print(f"Erreur : {e}")
    
    def acquisition(self):
        self.buttonlogin["state"] = DISABLED
        ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=1) # Connecte le capteur
        values = []
        time.sleep(2.5) # Attente de 2.5 secondes pour que l'utilisateur se positione.
        start_time = time.time()
        try:
            while time.time() - start_time < 10: # Créer une boucle pendant 10 secondes.
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip() # Lis les informations envoyer par le capteur
                    try:
                        current_time = time.time() - start_time
                        if current_time >= 2:  # Ignorer les 2 premières secondes (Les 2 première secondes ont souvent des parasites lors de l'acquisition)
                            values.append(float(line))
                    except ValueError:
                        pass  # Ignore les lignes non valides
        finally:
            ser.close()
        if not values == None or values == []:
            return values



    def autoconnect(self):
        if self.connected == True:
            self.logout()
        signal_client = self.acquisition()
        try: # Connection à la BDD
            db = mc.connect(host=self.readconfig()[2], user=self.readconfig()[3], password=self.readconfig()[4], database=self.readconfig()[5]) 
            cursor = db.cursor()
            var = "SELECT id_employe,signature_cardiaque FROM employes WHERE employes.signature_cardiaque IS NOT NULL;"
            cursor.execute(var) 
            results = cursor.fetchall()
            for i in range(0,len(results)): # Check chaque signature cardiaque pour pouvoir trouver une correspondante
                if len(signal_client) < 785:
                    self.error(8)
                    self.buttonlogin["state"] = NORMAL
                    return None
                pred = signaturecomparator.compare_signals(signal_client,list(map(float,results[i][1].replace(' ','').replace('[','').replace(']','').split(','))))
                print("Value pred : " + str(pred))
                if pred < 0.5: # Si la signature est bien la bonne
                    var = "SELECT type_serveur,Services.id_service,employes.nom FROM employes,Services,serveurs,droits WHERE employes.id_employe = " + str(results[i][0]) + " AND employes.id_service = Services.id_service AND droits.id_service = Services.id_service AND droits.id_serveur = serveurs.id_serveur"
                    cursor.execute(var)
                    results = cursor.fetchall()
                    self.listdroits.delete(0,END) # Réinitialise la Listbox
                    for i in range(0,len(results)):
                        self.listdroits.insert(i,results[i][0]) # Affiche la liste des droits
                    self.service = results[i][1]
                    self.id = results[i][2]
                    self.connect_to_serviceweb(self.id,self.service)
            self.error(9)
        except mc.Error as er: 
            print("Erreur de connexion à la base de données : {}".format(er))
            self.error(5)
            self.buttonlogin["state"] = NORMAL
        finally: 
            if cursor:
                cursor.close()
            if db:
                db.close()
            self.buttonlogin["state"] = NORMAL

    def changeripsweb(self): # Entre dans le fichier config les valeurs puis détruit la fenêtre 
        self.writeconfig(self.entryipsvweb.get(),self.entryportsvweb.get(),self.readconfig()[2],self.readconfig()[3],self.readconfig()[4],self.readconfig()[5])
        self.interfaceipchangesvweb.destroy()
        self.interfaceipchangesvweb.update()

    def changeripdbb(self): # Entre dans le fichier config les valeurs puis détruit la fenêtre 
        self.writeconfig(self.readconfig()[0],self.readconfig()[1],self.entryipbdd.get(),self.entryuserbdd.get(),self.entrymdpbdd.get(),self.entrybdd.get())
        self.interfaceipchangebdd.destroy()
        self.interfaceipchangebdd.update()

    def showmenuipchangesweb(self): # Détruit l'interface permettant de rentrer son mdp puis créer celle pour entrer les valeurs pour le fichier config
        if self.entrymdp2admin.get() == "Enigma":
            self.interfacemdp.destroy()
            self.interfacemdp.update()
            self.interfaceipchangesvweb = Toplevel(self)
            Label(self.interfaceipchangesvweb,text ="Quel adresse ip doit être utilisé pour le service web client ?").pack()
            self.entryipsvweb = Entry(self.interfaceipchangesvweb)
            self.entryipsvweb.pack()
            Label(self.interfaceipchangesvweb,text ="Quel port doit être utilisé pour le service web client ?").pack()
            self.entryportsvweb = Entry(self.interfaceipchangesvweb)
            self.entryportsvweb.pack()
            Button(self.interfaceipchangesvweb,
                text="Changez l'ip",
                fg="black",
                bg="lightgreen",
                activebackground="#21bf3b",
                command = self.changeripsweb).pack()
        else:
            self.error(6)

    def showmenuipchangebdd(self): # Détruit l'interface permettant de rentrer son mdp puis créer celle pour entrer les valeurs pour le fichier config
        if self.entrymdp2admin.get() == "Enigma":
            self.interfacemdp.destroy()
            self.interfacemdp.update()
            self.interfaceipchangebdd = Toplevel(self)
            Label(self.interfaceipchangebdd,text ="Quel adresse ip doit être utilisé pour la bdd ?").pack()
            self.entryipbdd = Entry(self.interfaceipchangebdd)
            self.entryipbdd.pack()
            Label(self.interfaceipchangebdd,text ="Quel base de donnée doit être utilisé ?").pack()
            self.entrybdd = Entry(self.interfaceipchangebdd)
            self.entrybdd.pack()
            Label(self.interfaceipchangebdd,text ="Quel utilisateur doit être utilisé ?").pack()
            self.entryuserbdd = Entry(self.interfaceipchangebdd)
            self.entryuserbdd.pack()
            Label(self.interfaceipchangebdd,text ="Quel mot de passe doit être utilisé ?").pack()
            self.entrymdpbdd = Entry(self.interfaceipchangebdd,show="*")
            self.entrymdpbdd.pack()
            Button(self.interfaceipchangebdd,text="Changez l'ip", fg="black", bg="lightgreen",activebackground="#21bf3b",command = self.changeripdbb).pack()
        else:
            self.error(6)

    def passwordinterface(self,redirect): # Créer l'interface permettant de saisir son mdp admin puis redirige a la bonne fonction.
        self.interfacemdp = Toplevel(self)
        self.interfacemdp.title("Mot de passe")
        Label(self.interfacemdp,text ="Insérer le mot de passe admin").pack()
        self.entrymdp2admin = Entry(self.interfacemdp,show="*")
        self.entrymdp2admin.pack()
        if redirect == 0:
            Button(self.interfacemdp,text="Se Connecter", fg="black", bg="lightgreen",activebackground="#21bf3b",command = self.showmenuipchangesweb).pack()
        if redirect == 1:
            Button(self.interfacemdp,text="Se Connecter", fg="black", bg="lightgreen",activebackground="#21bf3b",command = self.showmenuipchangebdd).pack()

    def connect_to_serviceweb(self,id,service): # Effectue la connection avec le service web , finalisation de la connexion
        link = "http://" + self.readconfig()[0] + ":" + self.readconfig()[1]
        self.clientweb = ClientServiceWeb.FirewallClient(server_url=link)
        if self.clientweb.login(id, self.getlocalip(),service):
            self.sendconnectioninfo("login") # Envoie un signal de connexion au broker
            self.connected = True
            tkinter.messagebox.showinfo(title="Success", message="Connection successful.") 
            self.buttonlogout['state'] = NORMAL # Permet la déconnexion
            
        else: self.error(4)

    def connect_to_database(self): # Effectue la connexion a la BDD et continue la procédure d'authentification.
        if self.connected == True:
            self.logout()
        db = None
        self.entryid["state"] = DISABLED
        self.entrymdp["state"] = DISABLED
        try:
            db = mc.connect(host=self.readconfig()[2], user=self.readconfig()[3], password=self.readconfig()[4], database=self.readconfig()[5]) 
            cursor = db.cursor()
            mdp = self.entrymdp.get()
            self.id = self.entryid.get()

            if mdp == "" or self.id == "": # Afficher une erreur si il n'y a pas d'identifiant/mdp valide
                self.error(1)
                return False

            var = 'SELECT type_serveur,Services.id_service FROM employes,Services,serveurs,droits WHERE employes.mdp = "' + mdp + '" AND employes.identifiant = "' + self.id + '" AND employes.id_service = Services.id_service AND droits.id_service = Services.id_service AND droits.id_serveur = serveurs.id_serveur'
            cursor.execute(var) # Effectue la requête à la BDD , les droits étant la seul information a recevoir.
            results = cursor.fetchall()
            if results == () :
                self.error(2) # Envoie un erreur si le compte n'est pas trouvé dans la BDD.
                self.entryid["state"] = NORMAL
                self.entrymdp["state"] = NORMAL
                return False
            self.service = results[0][1]
            self.listdroits.delete(0,END) # Réinitialise la Listbox
            for i in range(0,len(results)):
                self.listdroits.insert(i,results[i][0]) # Affiche la liste des droits
            
            else:
                self.connect_to_serviceweb(self.id,self.service) # Continue la procédure de connexion

        except mc.Error as er: 
            print("Erreur de connexion à la base de données : {}".format(er))
            self.error(5)
            self.entryid["state"] = NORMAL
            self.entrymdp["state"] = NORMAL
        finally: 
            if cursor:
                cursor.close()
            if db:
                db.close()
            self.entryid["state"] = NORMAL
            self.entrymdp["state"] = NORMAL

    def error(self,code): # Permet l'affichage des nombreuses boîtes d'erreur.
        if code == 1: tkinter.messagebox.showerror(title="Error", message="Erreur : Insérer votre mot de passe et nom d'utilisateur.")
        if code == 2: tkinter.messagebox.showerror(title="Error", message="Erreur : Le mot de passe ou utilisateur est invalide.")
        if code == 3: tkinter.messagebox.showerror(title="Error", message="Erreur : Erreur de connexion à la base de données , veuillez vérifier votre connexion ou l'état de la BDD.")
        if code == 4: tkinter.messagebox.showerror(title="Error", message="Erreur : Erreur de connexion au service web utilisateur , veuillez  vérifier votre connexion ou l'état du routeur.")
        if code == 5: tkinter.messagebox.showerror(title="Error", message="Erreur : La BDD n'est pas lancé.")
        if code == 6: tkinter.messagebox.showerror(title="Error", message="Erreur : Le mot de passe est invalide.")
        if code == 7: tkinter.messagebox.showerror(title="Error", message="Erreur : Erreur de connexion au capteur , veuillez vérifier son branchement.")
        if code == 8: tkinter.messagebox.showerror(title="Error", message="Erreur : Erreur de recognition, avez-vous bien porter le capteur ?")
        if code == 9: tkinter.messagebox.showerror(title="Error", message="Erreur : Votre signature cardiaque n'est pas reconnu , veuillez réessayer.")
        if code == 10: tkinter.messagebox.showerror(title="Error", message="Erreur : Impossible de se connecter au Broker.")


    def logout(self): # Permet la déconnexion dans le service web ainsi que l'envoie du signal de connexion au broker.
        self.listdroits.delete(0,END)
        self.clientweb.logout(self.id,self.getlocalip(),self.service)
        self.sendconnectioninfo("logout")
        self.buttonlogout['state'] = DISABLED
        self.connected = False
        self.service = None
        self.id = None
    
    def writeconfig(self,ipsw,portsw,ipdb,userdb,pswdb,nomdb): # Ecrit dans le fichier config.
        with open(self.config_file, 'w') as f:
            f.writelines(ipsw + '/' + portsw + '/' + ipdb + '/' + userdb + '/' + pswdb + '/' + nomdb)

    def readconfig(self): # Permet la lecture du fichier config , et défini les paramètres par défaut si celui-ci n'existe pas.
        if os.path.exists(self.config_file) == False:
            self.writeconfig("10.78.5.56","5000","10.78.5.123","projet","projet","projet_bdd")
        with open(self.config_file, 'r') as f:
            stringlist = f.readline().split('/')
            return stringlist


    def __init__(self): # Créer l'interface
        self.connected = False
        super().__init__()
        self.config_file = os.path.join(os.path.dirname(__file__), ".configihm.txt")
        self.clientweb = None
        self.title('Authenfication')
        self.menubar = Menu(self)
        self.menuadmin = Menu(self.menubar,tearoff = 0)
        self.menubar.add_cascade(label ='Admin', menu = self.menuadmin) 
        self.menuadmin.add_command(label="Changer l'ip du service web Client",command=lambda : self.passwordinterface(0))
        self.menuadmin.add_command(label="Changer l'ip de la BDD",command=lambda : self.passwordinterface(1))
        self.geometry("180x290+0+0")
        self.frme = Frame(self)
        self.frme.pack()

        self.labelconnect = Label(self.frme,text="Connexion")
        self.labelconnect.pack()

        self.labelid = Label(self.frme,text="Identifiant")
        self.labelid.pack(anchor="w",padx=20)

        self.entryid = Entry(self.frme)
        self.entryid.pack()

        self.labelmdp = Label(self.frme,text="Mot de passe",justify="left")
        self.labelmdp.pack(anchor="w",padx=20)

        self.entrymdp = Entry(self.frme,show="*")
        self.entrymdp.pack(pady=10)

        self.buttonloginauto = Button(self.frme,text="Connexion Automatique", fg="black", bg="lightblue",activebackground="#86bbbd", command=self.autoconnect)
        self.buttonloginauto.pack()
        self.buttonlogin = Button(self.frme,text="Se Connecter", fg="black", bg="lightgreen",activebackground="#60cc6b", command = self.connect_to_database)
        self.buttonlogin.pack()

        self.buttonlogout = Button(self.frme,text="Se Déconnecter",state=DISABLED, fg="black", bg="#ff757c",activebackground="#c94f55",command = self.logout)
        self.buttonlogout.pack()

        self.labeldroits = Label(self.frme,text="Droits Actuel :")
        self.labeldroits.pack(pady=10)
        self.listdroits = Listbox(self.frme,height=3)
        self.listdroits.pack()
        self.config(menu=self.menubar)
     
ihmauth = IHM()
ihmauth.resizable(width=False, height=False)
ihmauth.mainloop()
if not ihmauth.connected == True : # Si le client était connecté et que l'IHM est fermée , alors il sera automatiquement déconnecté.
    ihmauth.clientweb.logout()
