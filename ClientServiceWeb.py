import requests
from requests.exceptions import RequestException

class FirewallClient:
    """Classe qui gere la communication avec le serveur Flask."""

    def __init__(self, server_url):
        self.server_url = server_url

    def login(self, username, ip_address,user_rights):
        """Envoie les informations de connexion au serveur."""
        login_data = {"username": username, "ip_address": ip_address, "user_rights": user_rights}
        
        try:
            response = requests.post(f"{self.server_url}/login", json=login_data)
            response.raise_for_status()  # Verifie si la requete a echoue
            
            if response.status_code == 200:
                data = response.json()
                print("Connexion reussie.")
                print(f"Message: {data.get('message')}")
                return True
            else:
                print(f"Echec de la connexion : {response.json().get('message', 'Erreur inconnue')}")
                return False

        except RequestException as e:
            print(f"Erreur de connexion : {e}")
            return False
        
    def logout(self, username, ip_address,user_rights):
        """Envoie le signal de deconnexion au serveur."""
        logout_data = {"username": username, "ip_address": ip_address, "user_rights": user_rights}
        
        try:
            response = requests.post(f"{self.server_url}/logout", json=logout_data)
            response.raise_for_status()
            
            if response.status_code == 200:
                data = response.json()
                print("Deconnexion reussie.")
                print(f"Message: {data.get('message')}")
            else:
                print(f"Echec de la deconnexion : {response.json().get('message', 'Erreur inconnue')}")
        except RequestException as e:
            print(f"Erreur de deconnexion : {e}")

if __name__ == "__main__":
    client = FirewallClient(server_url="http://10.78.5.56:5000")

    # Test de connexion et configuration du pare-feu
    username = "user"
    ip_address = "10.78.5.114"  # Remplacez par l'adresse IP de l'utilisateur
    user_rights = "admin"
    #client.login(username, ip_address,user_rights)
        

        # Test de deconnexion
    #client.logout(username, ip_address,user_rights)
