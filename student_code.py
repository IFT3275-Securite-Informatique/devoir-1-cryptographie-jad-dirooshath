import random
import math
from collections import deque
import re

def decrypt(C):
    # Liste de symboles fixe
    symbols = [
        'b', 'j', '\r', 'J', '”', ')', 'Â', 'É', 'ê', '5', 't', '9', 'Y', '%', 'N', 'B', 'V', '\ufeff', 
        'Ê', '?', '’', 'i', ':', 's', 'C', 'â', 'ï', 'W', 'y', 'p', 'D', '—', '«', 'º', 'A', '3', 
        'n', '0', 'q', '4', 'e', 'T', 'È', '$', 'U', 'v', '»', 'l', 'P', 'X', 'Z', 'À', 'ç', 'u', 
        '…', 'î', 'L', 'k', 'E', 'R', '2', '_', '8', 'é', 'O', 'Î', '‘', 'a', 'F', 'H', 'c', '[', 
        '(', "'", 'è', 'I', '/', '!', ' ', '°', 'S', '•', '#', 'x', 'à', 'g', '*', 'Q', 'w', '1', 
        'û', '7', 'G', 'm', '™', 'K', 'z', '\n', 'o', 'ù', ',', 'r', ']', '.', 'M', 'Ç', '“', 'h', 
        '-', 'f', 'ë', '6', ';', 'd', 'ô', 'e ', 's ', 't ', 'es', ' d', '\r\n', 'en', 'qu', ' l', 
        're', ' p', 'de', 'le', 'nt', 'on', ' c', ', ', ' e', 'ou', ' q', ' s', 'n ', 'ue', 'an', 
        'te', ' a', 'ai', 'se', 'it', 'me', 'is', 'oi', 'r ', 'er', ' m', 'ce', 'ne', 'et', 'in', 
        'ns', ' n', 'ur', 'i ', 'a ', 'eu', 'co', 'tr', 'la', 'ar', 'ie', 'ui', 'us', 'ut', 'il', 
        ' t', 'pa', 'au', 'el', 'ti', 'st', 'un', 'em', 'ra', 'e,', 'so', 'or', 'l ', ' f', 'll', 
        'nd', ' j', 'si', 'ir', 'e\r', 'ss', 'u ', 'po', 'ro', 'ri', 'pr', 's,', 'ma', ' v', ' i', 
        'di', ' r', 'vo', 'pe', 'to', 'ch', '. ', 've', 'nc', 'om', ' o', 'je', 'no', 'rt', 'à ', 
        'lu', "'e", 'mo', 'ta', 'as', 'at', 'io', 's\r', 'sa', "u'", 'av', 'os', ' à', ' u', "l'", 
        "'a", 'rs', 'pl', 'é ', '; ', 'ho', 'té', 'ét', 'fa', 'da', 'li', 'su', 't\r', 'ée', 'ré', 
        'dé', 'ec', 'nn', 'mm', "'i", 'ca', 'uv', '\n\r', 'id', 'ni', 'bl'
    ]
    
    # Charger les fréquences des n-grammes
    def load_ngram_frequencies(filenames=['bigrammes_frequences.txt', 'trigrammes_frequences.txt',
                                         'tetragrammes_frequences.txt', 'pentagrammes_frequences.txt']):
        ngram_freq = {}
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        ngram = parts[0].upper()
                        frequency = float(parts[1])
                        ngram_freq[ngram] = frequency
        return ngram_freq
    
    # Charger le dictionnaire français
    def load_french_dictionary(filename='dictionnaire_francais.txt'):
        dictionary = set()
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip().lower()
                if word:
                    dictionary.add(word)
        return dictionary
    
    # Charger la liste des mots fréquents
    def load_frequent_words(filename='mots_frequents.txt'):
        frequent_words = set()
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip().lower()
                if word:
                    frequent_words.add(word)
        return frequent_words
    
    # Nettoyer le texte en supprimant la ponctuation et autres caractères non alphabétiques
    def clean_text(text):
        # Remplacer les caractères non alphabétiques par des espaces
        text = re.sub(r'[^a-zA-Zàâçéèêëîïôùûüÿñæœ\s]', ' ', text)
        # Remplacer les multiples espaces par un seul espace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    # Segmentation de texte en utilisant un algorithme simple de Maximum Matching
    def segment_text(text, dictionary, max_word_length):
        words = []
        index = 0
        
        while index < len(text):
            match = None
            for length in range(max_word_length, 0, -1):
                if index + length > len(text):
                    continue
                word = text[index:index+length]
                if word in dictionary:
                    match = word
                    break
            if match:
                words.append(match)
                index += len(match)
            else:
                words.append(text[index])
                index += 1
        return words
    
    # Calculer le score d'un texte déchiffré
    def get_score(text, ngram_freq, dictionary, frequent_words, max_word_length):
        score = 0
        min_freq = 1e-10  # Fréquence minimale pour les n-grammes inconnus
        ngram_weights = {2: 5, 3: 15, 4: 25, 5: 30}  # Pondérations des n-grammes
        
        # Score basé sur les n-grammes
        for n, weight in ngram_weights.items():
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n].upper()
                frequency = ngram_freq.get(ngram, min_freq)
                score += weight * math.log(frequency)
        
        # Nettoyer le texte pour une meilleure découpe des mots
        cleaned_text = clean_text(text.lower())
        words = segment_text(cleaned_text, dictionary, max_word_length)
        
        # Score basé sur les mots
        word_bonus = 0
        for word in words:
            if word in dictionary:
                if word in frequent_words:
                    word_bonus += 100  # Bonus pour les mots fréquents
                else:
                    word_bonus += 500  # Bonus pour les mots du dictionnaire
            else:
                word_bonus -= 25  # Pénalité pour les mots non valides
        score += word_bonus
        
        return -score  # On prend l'opposé pour minimiser le score
    
    # Générer une clé de déchiffrement aléatoire
    def generate_random_mapping(binary_strings, symbols):
        mapping = {}
        shuffled_symbols = symbols[:]
        random.shuffle(shuffled_symbols)
        for i, bs in enumerate(binary_strings):
            if i < len(shuffled_symbols):
                mapping[bs] = shuffled_symbols[i]
            else:
                # Si plus de binary_strings que de symbols, assigner des symboles aléatoires
                mapping[bs] = random.choice(symbols)
        return mapping
    
    # Déchiffrer le cryptogramme avec la clé actuelle
    def decrypt_with_mapping(binary_strings, mapping):
        decrypted_text = ''.join([mapping.get(bs, '?') for bs in binary_strings])
        return decrypted_text
    
    # Générer une clé voisine en utilisant une attaque par distance
    def get_neighbor_mapping(current_mapping, binary_strings, ngram_freq, dictionary, frequent_words, max_word_length, num_swap_candidates=100):
        best_swap = None
        best_swap_score = float('inf')
        
        for _ in range(num_swap_candidates):
            bs1, bs2 = random.sample(binary_strings, 2)
            if current_mapping[bs1] == current_mapping[bs2]:
                continue  # Éviter d'échanger des symboles identiques
            new_mapping = current_mapping.copy()
            new_mapping[bs1], new_mapping[bs2] = new_mapping[bs2], new_mapping[bs1]
            
            decrypted_neighbor = decrypt_with_mapping(binary_strings, new_mapping)
            swap_score = get_score(decrypted_neighbor, ngram_freq, dictionary, frequent_words, max_word_length)
            
            if swap_score < best_swap_score:
                best_swap_score = swap_score
                best_swap = new_mapping
        
        if best_swap:
            return best_swap
        else:
            # Si aucun swap n'améliore le score, retourner un swap aléatoire
            return get_random_swap(current_mapping, binary_strings)
    
    # Générer une clé voisine en échangeant deux mappings aléatoirement
    def get_random_swap(current_mapping, binary_strings):
        new_mapping = current_mapping.copy()
        bs1, bs2 = random.sample(binary_strings, 2)
        new_mapping[bs1], new_mapping[bs2] = new_mapping[bs2], new_mapping[bs1]
        return new_mapping
    
    # Fonction principale de Tabu Search
    def tabu_search(cryptogram, ngram_freq, dictionary, frequent_words, symbols, 
                   tabu_size=500, max_iterations=500000, num_neighbors=200):
        if len(cryptogram) % 8 != 0:
            raise ValueError("Le cryptogramme doit être une chaîne de longueur multiple de 8.")
        
        binary_strings = [cryptogram[i:i+8] for i in range(0, len(cryptogram), 8)]
        unique_bs = list(set(binary_strings))
        
        if len(unique_bs) > len(symbols):
            raise ValueError("Nombre de binary strings uniques dépasse le nombre de symboles disponibles.")
        
        # Précompute max_word_length avec une limite pour éviter les calculs longs
        if dictionary:
            max_word_length = max(len(word) for word in dictionary)
            max_word_length = min(max_word_length, 20)  # Limite à 20 pour des performances optimales
        else:
            max_word_length = 10
        
        # Initialiser la clé de déchiffrement aléatoire
        current_mapping = generate_random_mapping(unique_bs, symbols)
        decrypted_text = decrypt_with_mapping(binary_strings, current_mapping)
        current_score = get_score(decrypted_text, ngram_freq, dictionary, frequent_words, max_word_length)
        
        best_mapping = current_mapping.copy()
        best_score = current_score
        
        # Initialiser la liste taboue
        tabu_list = deque(maxlen=tabu_size)
        
        for iteration in range(1, max_iterations + 1):
            # Générer des mappings voisins en utilisant une attaque par distance
            neighbors = []
            for _ in range(num_neighbors):
                neighbor = get_neighbor_mapping(current_mapping, unique_bs, ngram_freq, dictionary, frequent_words, max_word_length)
                neighbors.append(neighbor)
            
            # Évaluer tous les voisins et choisir le meilleur non tabou
            best_neighbor = None
            best_neighbor_score = float('inf')
            for neighbor in neighbors:
                neighbor_key = tuple(sorted(neighbor.items()))
                if neighbor_key in tabu_list:
                    continue
                decrypted_neighbor = decrypt_with_mapping(binary_strings, neighbor)
                neighbor_score = get_score(decrypted_neighbor, ngram_freq, dictionary, frequent_words, max_word_length)
                if neighbor_score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score
            
            # Si aucun voisin non tabou n'est trouvé, continuer
            if best_neighbor is None:
                continue
            
            # Accepter le meilleur voisin
            current_mapping = best_neighbor
            current_score = best_neighbor_score
            
            # Mettre à jour la liste taboue
            neighbor_key = tuple(sorted(current_mapping.items()))
            tabu_list.append(neighbor_key)
            
            # Mettre à jour le meilleur mapping trouvé
            if current_score < best_score:
                best_mapping = current_mapping.copy()
                best_score = current_score
                decrypted_best = decrypt_with_mapping(binary_strings, best_mapping)
                print(f"Iteration {iteration}, Best Score: {best_score}")
                print(f"Décryptage partiel : {decrypted_best[:200]}")
                print('--------')
        
        # Déchiffrer avec la meilleure clé trouvée
        final_decrypted_text = decrypt_with_mapping(binary_strings, best_mapping)
        return final_decrypted_text

    # Charger les fréquences des n-grammes
    ngram_filenames = [
        'bigrammes_frequences.txt',
        'trigrammes_frequences.txt',
        'tetragrammes_frequences.txt',
        'pentagrammes_frequences.txt'
    ]
    ngram_freq = load_ngram_frequencies(ngram_filenames)
    
    # Charger le dictionnaire français et les mots fréquents
    dictionary = load_french_dictionary('dictionnaire_francais.txt')
    frequent_words = load_frequent_words('mots_frequents.txt')
    
    # Appeler la fonction de décryptage
    decrypted_message = tabu_search(
        cryptogram=C,
        ngram_freq=ngram_freq,
        dictionary=dictionary,
        frequent_words=frequent_words,
        symbols=symbols,
        tabu_size=500,
        max_iterations=250000,
        num_neighbors=200
    )

    return decrypted_message
