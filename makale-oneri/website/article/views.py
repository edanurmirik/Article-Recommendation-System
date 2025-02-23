import glob
import json
from bson import ObjectId
from django.shortcuts import render, redirect
from django.http.response import HttpResponse
from django.contrib import messages
import pymongo
import os
import re
from typing import Counter
import nltk
import fasttext
import string
from nltk.corpus import stopwords
import numpy as np
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from django.http import JsonResponse


nltkIslemleribool=1
fastTextIslemleribool=1
sciBertIslemleribool=1
print_embeddingsbool=1
ilgiAlanlariBelirleme=1
process_text_filesBool=1

islenmemisInspec = "inspec"
islenmisInspec = "temizlenmis_inspec"


db_url="mongodb+srv://edanrmirik:755502Eda@cluster0.nz8jxm9.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(db_url)
db = client.yazlab

def index(request):  
    return render(request, "index.html") 

def login_page(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        success = custom_login(request, email, password)
        print(success)

        if success:
            user = db.user_information.find_one({'email': email})
            
            request.session['_id'] = str(user['_id']) 
            request.session['name'] = user['name']
            request.session['email'] = user['email']

            return redirect('home')
        else:
            messages.error(request, 'Geçersiz e-posta adresi veya şifre.')
            return render(request, 'login_page.html')

    return render(request, 'login_page.html')

def signup(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')

        existing_person = db.user_information.find_one({'email': email})
        if existing_person:
            messages.warning(request, 'Bu e-posta adresi zaten kullanımda. Giriş yapmayı deneyin.')
            return render(request, 'signup.html')
        else: 
            try:
                db.user_information.insert_one({
                    'name': name,
                    'email': email,
                    'password': password
                })
                user = db.user_information.find_one({'email': email})

                request.session['_id'] = str(user['_id'])
                request.session['name'] = user['name']
                request.session['email'] = user['email']
                
                return redirect('interests')

            except Exception as e:
                return HttpResponse("hatalı")

    return render(request, 'signup.html')

def home(request):
    name = request.session.get('name')
    user_id = request.session.get('_id')

    ilgiAlanlarii = list(db.user_interests.find({'user_id': user_id}, {'_id': 0, 'interests': 1}))
    ilgiAlanlari= ilgiAlanlarii[0]['interests']

    user_interests = list(db.user_interests_article.find({'user_id':user_id}))
    article_names = []
    if user_interests:
        for interest in user_interests:
            if 'similarity_data' in interest:
                article_names.extend([article['name'] for article in interest['similarity_data'] if article.get('state') == 1])
    else:
        article_names = []

    directory = "temizlenmis_inspec"
    file_contents = []
    
    for file_name in article_names:
        content = find_and_read_file(directory, file_name)
        if content:
            file_contents.append(content)
        else:
            print(f"{file_name} adlı dosya bulunamadı.")

    file_content = [t.replace('\n', '').replace('\t', '') for t in file_contents]
    
    begenilereGoreVektor = []
    for ilgi in ilgiAlanlari:
        begenilereGoreVektor.append(ilgi)

    for filec in file_content:
       begenilereGoreVektor.append(filec) 
     
    """model_path = "cc.en.300.bin"
    word_list = begenilereGoreVektor
    average_vector_fastText = calculate_average_vector_for_fastText(word_list, model_path)
    fastTextIslemleri(model_path, average_vector_fastText,user_id,article_names)
    #print(average_vector_fastText)

    modelYoluu = "allenai/scibert_scivocab_uncased"
    word_listt = begenilereGoreVektor
    average_vector_sciBert = calculate_average_vector_for_SciBERT(word_listt, modelYoluu)
    find_most_similar_articles_SciBERT(average_vector_sciBert, modelYoluu,user_id)
    #print(average_vector_sciBert)"""

    fasttext_articles_data = db.user_interests_article.find_one({'user_id': user_id, 'tur': 'fastText'})
    scibert_articles_data = db.user_interests_article.find_one({'user_id': user_id, 'tur': 'sciBert'})

    fasttext_article_names = [article['name'] for article in fasttext_articles_data['similarity_data']] if fasttext_articles_data else []
    scibert_article_names = [article['name'] for article in scibert_articles_data['similarity_data']] if scibert_articles_data else []

    
    fasttext_articles = list(db.article_information.find({'filename': {'$in': fasttext_article_names}}))
    scibert_articles = list(db.article_information.find({'filename': {'$in': scibert_article_names}}))

    for article in fasttext_articles:
        article['mongo_id'] = article['_id'] 
    for article in scibert_articles:
        article['mongo_id'] = article['_id']

    precision_degeri=precision_hesapla(user_id)

    return render(request, 'home.html', {
        'name': name,
        'fasttext_articles': fasttext_articles,
        'scibert_articles': scibert_articles,
        'precision_degeri':precision_degeri
        })

def interests(request):
    interests = db.interests.find()    
    if request.method == 'POST':
        user_interest = request.POST.getlist('interest')
        user_id = request.session.get('_id') 

        db.user_interests.insert_one({
            'user_id': user_id,
            'interests': user_interest
        })
        print(user_interest)

        model_path = "cc.en.300.bin"
        word_list = user_interest
        average_vector_fastText = calculate_average_vector_for_fastText(word_list, model_path)
        ilgiAlanlari_fastTextIslemleri(islenmisInspec, model_path, average_vector_fastText,user_id)
        #print(average_vector_fastText)

        modelYoluu = "allenai/scibert_scivocab_uncased"
        word_listt = user_interest
        average_vector_sciBert = calculate_average_vector_for_SciBERT(word_listt, modelYoluu)
        ilgiAlanlari_find_most_similar_articles_SciBERT(average_vector_sciBert, modelYoluu,user_id)
        #print(average_vector_sciBert)
        
 
        return redirect('home')

    return render(request, 'interests.html', {'interests': interests})

def logout_page(request):

    if 'name' in request.session:
        del request.session['name']
        del request.session['_id']

    return redirect('index')

def profil_page(request):
    user_email = request.session.get('email')
    user = db.user_information.find_one({'email': user_email})

    return render(request, 'profil_page.html', {'user': user})

def update_profil(request):
    user_email = request.session.get('email')
    user_id = request.session.get('_id')
    user = db.user_information.find_one({'email': user_email})

    if request.method == 'POST':
        if 'update' in request.POST:
            name = request.POST.get('name')
            yeni_email = request.POST.get('email')
            yeni_password = request.POST.get('password')

            db.user_information.update_one({'email': user_email}, {'$set': {'name': name, 'email': yeni_email, 'password': yeni_password}})
        
            request.session['email'] = yeni_email
            request.session['name'] = name
            request.session.save()

            messages.success(request, 'Profil bilgileriniz başarıyla güncellendi.')
            return redirect('profil_page')

        elif 'delete' in request.POST:
            db.user_information.delete_one({'email':user_email})
            db.user_interests.delete_one({'user_id':user_id})
            db.algorithm_for_fastText.delete_one({'user_id':user_id})
            db.algorithm_for_sciBert.delete_one({'user_id':user_id})
            db.personalised_articles.delete_many({'user_id':user_id})
            db.user_interests_article.delete_many({'user_id':user_id})
            db.viewed.delete_many({'user_id':user_id})

            messages.success(request, 'Kaydınız başarıyla silindi.')
            return redirect('signup')

    return render(request, 'profil_page.html', {'user': user})

def articles(request):
    articles = list(db.article_information.find())

    for article in articles:
        article['mongo_id'] = article['_id']

    return render(request, 'articles.html', {'articles': articles})

def update_interests(request):
    user_id = request.session.get('_id')
    user_interests = db.user_interests.find_one({'user_id': user_id})

    interests = db.interests.find()

    user_selected_interests = request.POST.getlist('interest')

    if request.method == 'POST':
        db.user_interests.update_one({'user_id': user_id}, {'$set': {'interests': user_selected_interests}})

        messages.success(request, 'İlgi alanı bilgileriniz başarıyla güncellendi.')
        return redirect('update_interests')

    return render(request, 'update_interests.html', {'interests': interests})

def custom_login(request, email=None, password=None):
    try:
        user = db.user_information.find_one({'email': email})

        if not user or user['password'] != password:
            print(user['password'])
            print(password)
            return False
        else:
            return True

    except Exception as e:
        return None
    
def toggle_like_dislike(request):

    if request.method == 'POST':
        data = json.loads(request.body)
        article_id = data.get('article_id')
        action = data.get('action')
        user_id = request.session.get('_id')

        article = db.article_information.find_one({'_id' : ObjectId(article_id)})

        if article and action and user_id:
            query = {'user_id': user_id, 'similarity_data.name': article.get('filename')}
            update_query = {'$set': {'similarity_data.$[element].state': 1 if action == 'like' else -1}}

            db.user_interests_article.update_one(
                query,
                update_query,
                upsert=True,
                array_filters=[{'element.name': article.get('filename')}]
            )
            return JsonResponse({'status': 'success'})

    return JsonResponse({'status': 'error'}, status=400)
    
def liked_articles(request):
    user_id = request.session.get('_id')

    user_interests = list(db.user_interests_article.find({'user_id':user_id}))

    article_names = []
    if user_interests:
        for interest in user_interests:
            if 'similarity_data' in interest:
                article_names.extend([article['name'] for article in interest['similarity_data'] if article.get('state') == 1])
    else:
        article_names = []


    if article_names:
        liked_articles = list(db.article_information.find({'filename': {'$in': article_names}}))
    else:
        liked_articles = []

    for article in liked_articles:
        article['mongo_id'] = article['_id']

    return render(request, "liked_articles.html", {'articles': liked_articles})

def article_detail(request,mongo_id):
    article = db.article_information.find_one({'_id' : ObjectId(mongo_id)})

    filename = article.get('filename')  
    
   
    artic = db.user_interests_article.find_one({'similarity_data.name': filename})
    
    
    rate = None
    if artic:
        similarity_data = artic.get('similarity_data', [])
        for data in similarity_data:
            if data.get('name') == filename:
                user_id = request.session.get('_id')
                rate = data.get('rate')
                db.viewed.insert_one({
                'user_id':user_id,
                'name':data.get('name')
                })
                break

    return render(request, "article_detail.html", {'article':article,'rate':rate})

def fastTextVectorKaydet(islenmisInspec, modelYolu):
   
    model = fasttext.load_model(modelYolu)

    
    file_names = []
    vector_representations = []

    
    for file_name in os.listdir(islenmisInspec):
        file_path = os.path.join(islenmisInspec, file_name)
        with open(file_path, "r") as file:
            content = file.read()
        
       
        content = content.replace('\n', ' ')

        
        vector_representation = model.get_sentence_vector(content)
        vector_representation = vector_representation.tolist()

        db.fastTextVectors.insert_one({
                    'file_name':file_name,
                    'vector_representation': vector_representation
                })

def sciBertVectorKaydet(folder_path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

   
    max_length = 512
    tokenizer.padding_side = "right"  

    
    embeddings = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                content = file.read()
            
            
            content = content[:max_length]

            inputs = tokenizer(content, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            embeddings[file_name] = last_hidden_state

            db.sciBertVectors.insert_one({
                    'file_name':file_name,
                    'vector_representation': last_hidden_state.tolist()
                })

def nltkIslemleri(islenmemisInspec, islenmisInspec):
    nlp = spacy.load("en_core_web_sm")
    
    stop_words = nlp.Defaults.stop_words
    punctuation = set(string.punctuation)

    if not os.path.exists(islenmisInspec):
        os.makedirs(islenmisInspec)

    for file_name in os.listdir(islenmemisInspec):
        if file_name.endswith(".txt"):
            file_path = os.path.join(islenmemisInspec, file_name)
            
            with open(file_path, "r") as file:
                content = file.read()

            doc = nlp(content)
            clean_words = [token.lemma_.lower() for token in doc if token.lemma_.lower() not in stop_words and token.lemma_.lower() not in punctuation]

            clean_content = ' '.join(clean_words)

            new_file_path = os.path.join(islenmisInspec, file_name)
            with open(new_file_path, "w") as new_file:
                
                new_file.write(clean_content)

            print(f"Temizlenmiş içerik {file_name} yeni klasöre kaydedildi: {new_file_path}")

def ilgiAlanlari_fastTextIslemleri(islenmisInspec, modelYolu, search_word_vector,user_id ):
        
    model = fasttext.load_model(modelYolu)

   
    file_names = []
    vector_representations = []
    similarty_names =[]
    similarty_rates =[]
        
    cursor = db.fastTextVectors.find()
    
    for document in cursor:
        file_names.append(document['file_name'])
        vector_representations.append(document['vector_representation'])

   

    
    vectors_np = np.array(vector_representations)

    
    search_word_vector_np = np.array([search_word_vector])

    
    similarity_scores = cosine_similarity(search_word_vector_np, vectors_np)

    
    top_matching_indices = np.argsort(similarity_scores[0])[-5:][::-1]

    
    print("\nEn çok eşleşen dosyalar:")
    for index in top_matching_indices:
        print(f"Dosya Adı: {file_names[index]}, Benzerlik Skoru: {similarity_scores[0][index]}")
        similarty_names.append(file_names[index])
        similarty_rates.append(similarity_scores[0][index])


    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate, in zip(similarty_names, similarty_rates)]
    
    db.user_interests_article.insert_one({
    'user_id': user_id,
    'similarity_data': similarity_data,
    'tur': 'fastText'
    })

def ilgiAlanlari_find_most_similar_articles_SciBERT(average_vector_sciBert, model_name,user_id):
 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    similarty_names =[]
    similarty_rates =[]
    search_embedding = average_vector_sciBert

    
    article_embeddings = {}
    for article in db.sciBertVectors.find():
        file_name = article['file_name']
        vector_representation = np.array(article['vector_representation'])
        article_embeddings[file_name] = vector_representation

    
    similarities = {}
    for file_name, vector_representation in article_embeddings.items():
        similarity = cosine_similarity([search_embedding], [vector_representation])[0][0]
        similarities[file_name] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    most_similar_articles = [article[0] for article in sorted_similarities[:5]]

    
    print("En uygun 5 makale:")
    for article, similarity in sorted_similarities[:5]:
        print(f"Makale Adı: {article}, Benzerlik Oranı: {similarity}")
        similarty_names.append(article)
        similarty_rates.append(similarity)
        
    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate in zip(similarty_names, similarty_rates)]    
    db.user_interests_article.insert_one({
    'user_id': user_id,
    'similarity_data': similarity_data,
    'tur': 'sciBert'
    })

    return most_similar_articles

def read_files_in_folder(folder_path):
    all_words = []

    
    for filename in os.listdir(folder_path):
        if filename.endswith(".key"):  
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                
                for line in file:
                    line = line.strip().lower()  
                    if line:  
                        all_words.append(line)

    return all_words

def calculate_average_vector_for_fastText(text_list, model_path):
 
   
    try:
        model = fasttext.load_model(model_path)
    except ValueError as e:
        print("Model yüklenirken bir hata oluştu:", e)
        return None

   
    if not model:
        print("Model yüklenemedi.")
        return None

    
    text_vectors = [model.get_sentence_vector(text) for text in text_list]

    
    average_vector = np.mean(text_vectors, axis=0)
    
    return average_vector

def calculate_average_vector_for_SciBERT(text_list, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    text_embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        text_embeddings.append(last_hidden_state)

    average_vector = np.mean(text_embeddings, axis=0)
    
    return average_vector

def process_text_files(cleaned_dir, keys_dir):

    titles = []
    summaries = []
    filenames = []


    for filename in os.listdir(cleaned_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(cleaned_dir, filename), "r", encoding="utf-8") as file:
                first_line = file.readline().strip()  
                titles.append(first_line)  
                summary = file.read() 
                summaries.append(summary)  
                filenames.append(filename)  

    
    for title, summary, filename in zip(titles, summaries, filenames):
        
        key_filename = os.path.splitext(filename)[0] + ".key"  
        key_filepath = os.path.join(keys_dir, key_filename)
        if os.path.exists(key_filepath):
            with open(key_filepath, "r", encoding="utf-8") as key_file:
                key_content = key_file.read().strip()  
        else:
            key_content = "" 

   
        db.article_information.insert_one({
            "title": title,
            "summary": summary,
            "filename": filename,
            "key_content": key_content
        })

def find_and_read_file(directory, file_name):
   
    search_pattern = os.path.join(directory, f"{file_name}")
    matching_files = glob.glob(search_pattern)

    
    if matching_files:
        file_path = matching_files[0]  
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    else:
        return None

def fastTextIslemleri(modelYolu, search_word_vector,user_id, article_names,unliked_article_names):
       
    model = fasttext.load_model(modelYolu)

    
    file_names = []
    vector_representations = []
    similarty_names =[]
    similarty_rates =[]
        
    cursor = db.fastTextVectors.find()
   
    for document in cursor:
        file_names.append(document['file_name'])
        vector_representations.append(document['vector_representation'])


    vectors_np = np.array(vector_representations)

    search_word_vector_np = np.array([search_word_vector])

    similarity_scores = cosine_similarity(search_word_vector_np, vectors_np)


    top_matching_indices = np.argsort(similarity_scores[0])[-10:][::-1]


    """print("\nEn çok eşleşen dosyalar:")
    for index in top_matching_indices:
        print(f"Dosya Adı: {file_names[index]}, Benzerlik Skoru: {similarity_scores[0][index]}")
        similarty_names.append(file_names[index])
        similarty_rates.append(similarity_scores[0][index])"""


    for index in top_matching_indices:

        if file_names[index] in unliked_article_names:
            continue

        similarty_names.append(file_names[index])
        similarty_rates.append(similarity_scores[0][index])

        if len(similarty_names) == 5:
            break

    print("\nEn çok eşleşen dosyalar:")
    for name, score in zip(similarty_names, similarty_rates):
        print(f"Dosya Adı: {name}, Benzerlik Skoru: {score}")


    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate, in zip(similarty_names, similarty_rates)]
    
    db.user_interests_article.delete_one({
    'user_id': user_id,
    'tur': 'fastText'
    })

    db.user_interests_article.insert_one({
    'user_id': user_id,
    'similarity_data': similarity_data,
    'tur': 'fastText'
    })

def find_most_similar_articles_SciBERT(average_vector_sciBert, model_name,user_id,unliked_article_names):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    similarty_names =[]
    similarty_rates =[]
    search_embedding = average_vector_sciBert

    article_embeddings = {}
    for article in db.sciBertVectors.find():
        file_name = article['file_name']
        vector_representation = np.array(article['vector_representation'])
        article_embeddings[file_name] = vector_representation

    similarities = {}
    for file_name, vector_representation in article_embeddings.items():
        similarity = cosine_similarity([search_embedding], [vector_representation])[0][0]
        similarities[file_name] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    most_similar_articles = []
    for article, similarity in sorted_similarities:
        if article not in unliked_article_names:
            most_similar_articles.append((article, similarity))
        if len(most_similar_articles) == 5:
            break

    print("En uygun 5 makale:")
    for article, similarity in most_similar_articles:
        print(f"Makale Adı: {article}, Benzerlik Oranı: {similarity}")
        similarty_names.append(article)
        similarty_rates.append(similarity)
        
    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate in zip(similarty_names, similarty_rates)]    

    db.user_interests_article.delete_one({
        'user_id': user_id,
        'tur': 'sciBert'
        })

    db.user_interests_article.insert_one({
    'user_id': user_id,
    'similarity_data': similarity_data,
    'tur': 'sciBert'
    })

    return most_similar_articles

def precision_hesapla(user_id):

    TP = 0
    FP =0 

    ilgiAlanlarii = list(db.user_interests.find({'user_id': user_id}, {'_id': 0, 'interests': 1}))
    ilgiAlanlari= ilgiAlanlarii[0]['interests']

    user_interests = list(db.user_interests_article.find({'user_id':user_id}))
    article_names = []
    liked_article_names = []
    unliked_article_names = []
    if user_interests:
        for interest in user_interests:
            if 'similarity_data' in interest:
                liked_article_names.extend([article['name'] for article in interest['similarity_data'] if article.get('state') == 1])
                unliked_article_names.extend([article['name'] for article in interest['similarity_data'] if article.get('state') == -1])
                article_names.extend([article['name'] for article in interest['similarity_data']])
    else:
        article_names = []
        unliked_article_names = []

    print(ilgiAlanlari)
    article_namess = [int(name.replace('.txt', '')) for name in article_names]
    print(article_namess)

    keys_folder = 'keys'  

    for article_id in article_namess:
        key_file_path = os.path.join(keys_folder, f"{article_id}.key")
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r') as file:
                keys = file.read().splitlines()
                if any(interest in keys for interest in ilgiAlanlari):
                    print(f"Makale ID: {article_id}, Eşleşme Durumu: İlgi alanı ile eşleşiyor")
                    TP=TP+1
                else:
                    print(f"Makale ID: {article_id}, Eşleşme Durumu: İlgi alanı ile eşleşmiyor")
                    FP=FP+1
        else:
            print(f"Makale ID: {article_id}, Eşleşme Durumu: Anahtar kelime dosyası bulunamadı")
    
    precision_degeri = TP/(TP+FP)
    print(precision_degeri)

    return precision_degeri

def ikinci_asama(request):
    name = request.session.get('name')
    user_id = request.session.get('_id')

    ilgiAlanlarii = list(db.user_interests.find({'user_id': user_id}, {'_id': 0, 'interests': 1}))
    ilgiAlanlari= ilgiAlanlarii[0]['interests']

    user_interests = list(db.user_interests_article.find({'user_id':user_id}))
    article_names = []
    unliked_article_names = []
    if user_interests:
        for interest in user_interests:
            if 'similarity_data' in interest:
                article_names.extend([article['name'] for article in interest['similarity_data'] if article.get('state') == 1])
                unliked_article_names.extend([article['name'] for article in interest['similarity_data'] if article.get('state') == -1])
    else:
        article_names = []
        unliked_article_names = []

    directory = "temizlenmis_inspec"  
    file_contents = []

    print(unliked_article_names)
    
    for file_name in article_names:
        content = find_and_read_file(directory, file_name)
        if content:
            file_contents.append(content)
        else:
            print(f"{file_name} adlı dosya bulunamadı.")

    file_content = [t.replace('\n', '').replace('\t', '') for t in file_contents]
    
    begenilereGoreVektor = []
    for ilgi in ilgiAlanlari:
        begenilereGoreVektor.append(ilgi)

    for filec in file_content:
       begenilereGoreVektor.append(filec)
    model_path = "cc.en.300.bin"
    word_list = begenilereGoreVektor
    average_vector_fastText = calculate_average_vector_for_fastText(word_list, model_path)
    fastTextIslemleri(model_path, average_vector_fastText,user_id,article_names,unliked_article_names)
    #print(average_vector_fastText)

    modelYoluu = "allenai/scibert_scivocab_uncased"
    word_listt = begenilereGoreVektor
    average_vector_sciBert = calculate_average_vector_for_SciBERT(word_listt, modelYoluu)
    find_most_similar_articles_SciBERT(average_vector_sciBert, modelYoluu,user_id,unliked_article_names)
    #print(average_vector_sciBert)"""

    fasttext_articles_data = db.user_interests_article.find_one({'user_id': user_id, 'tur': 'fastText'})
    scibert_articles_data = db.user_interests_article.find_one({'user_id': user_id, 'tur': 'sciBert'})

    fasttext_article_names = [article['name'] for article in fasttext_articles_data['similarity_data']] if fasttext_articles_data else []
    scibert_article_names = [article['name'] for article in scibert_articles_data['similarity_data']] if scibert_articles_data else []

    # Makale detaylarını articles koleksiyonundan çek
    fasttext_articles = list(db.article_information.find({'filename': {'$in': fasttext_article_names}}))
    scibert_articles = list(db.article_information.find({'filename': {'$in': scibert_article_names}}))

    for article in fasttext_articles:
        article['mongo_id'] = article['_id'] 
    for article in scibert_articles:
        article['mongo_id'] = article['_id']

    precision_degeri=precision_hesapla(user_id)

    return render(request, 'home.html', {
        'name': name,
        'fasttext_articles': fasttext_articles,
        'scibert_articles': scibert_articles,
        'precision_degeri':precision_degeri
        })

def algoritma_oneriler(request):
    user_id = request.session.get('_id')
    name = request.session.get('_id')
    db_beğenilmisler = list(db.viewed.find({'user_id': user_id}, {'_id': 0, 'name': 1}))
    db_ilgiler = list(db.user_interests.find({'user_id': user_id}, {'_id': 0, 'interests': 1}))
    ozetler = []

    ilgiler = db_ilgiler[0]['interests']

    for ilgi in ilgiler:
        print(ilgi)
        ozetler.append(ilgi)
    
    for begenilmis_makale in db_beğenilmisler:
        makale_ismi = begenilmis_makale['name']
    
   
        makale_bilgisi = db.article_information.find_one({'filename': makale_ismi}, {'_id': 0,'summary': 1})
        #print(makale_bilgisi)
        ozetler.append(makale_bilgisi['summary'])

    ozetler_duzenlenmis = [ozet.replace('\n', '').replace('\t', '') for ozet in ozetler]

    print(ozetler_duzenlenmis)

    model_path = "cc.en.300.bin"
    word_list = ozetler_duzenlenmis
    average_vector_fastText = calculate_average_vector_for_fastText(word_list, model_path)
    algoritma_fastText(average_vector_fastText,user_id)
    #print(average_vector_fastText)

    modelYoluu = "allenai/scibert_scivocab_uncased"
    word_listt = ozetler_duzenlenmis
    average_vector_sciBert = calculate_average_vector_for_SciBERT(word_listt, modelYoluu)
    algoritma_sciBert(average_vector_sciBert, modelYoluu,user_id)
    #print(average_vector_sciBert)

    fasttext_articles_data = db.algorithm_for_fastText.find_one({'user_id': user_id, 'tur': 'fastText'})
    scibert_articles_data = db.algorithm_for_sciBert.find_one({'user_id': user_id, 'tur': 'sciBert'})

    fasttext_article_names = [article['name'] for article in fasttext_articles_data['similarity_data']] if fasttext_articles_data else []
    scibert_article_names = [article['name'] for article in scibert_articles_data['similarity_data']] if scibert_articles_data else []

   
    fasttext_articles = list(db.article_information.find({'filename': {'$in': fasttext_article_names}}))
    scibert_articles = list(db.article_information.find({'filename': {'$in': scibert_article_names}}))

    for article in fasttext_articles:
        article['mongo_id'] = article['_id'] 
    for article in scibert_articles:
        article['mongo_id'] = article['_id']

    return render(request, 'algorithm_articles.html', {
        'name': name,
        'fasttext_articles': fasttext_articles,
        'scibert_articles': scibert_articles,
        })

def algoritma_fastText(search_word_vector,user_id):
   
    file_names = []
    vector_representations = []
    similarty_names =[]
    similarty_rates =[]
        
    cursor = db.fastTextVectors.find()

    for document in cursor:
        file_names.append(document['file_name'])
        vector_representations.append(document['vector_representation'])

    vectors_np = np.array(vector_representations)


    search_word_vector_np = np.array([search_word_vector])


    similarity_scores = cosine_similarity(search_word_vector_np, vectors_np)


    top_matching_indices = np.argsort(similarity_scores[0])[-5:][::-1]


    print("\nEn çok eşleşen dosyalar:")
    for index in top_matching_indices:
        print(f"Dosya Adı: {file_names[index]}, Benzerlik Skoru: {similarity_scores[0][index]}")
        similarty_names.append(file_names[index])
        similarty_rates.append(similarity_scores[0][index])


    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate, in zip(similarty_names, similarty_rates)]
    
    db.algorithm_for_fastText.update_one(
        {'user_id': user_id},  
        {
            '$set': {
                'similarity_data': similarity_data,
                'tur': 'fastText'
            }
        },
        upsert=True  
    )  

def algoritma_sciBert(average_vector_sciBert, model_name,user_id):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    similarty_names =[]
    similarty_rates =[]
    search_embedding = average_vector_sciBert


    article_embeddings = {}
    for article in db.sciBertVectors.find():
        file_name = article['file_name']
        vector_representation = np.array(article['vector_representation'])
        article_embeddings[file_name] = vector_representation

    similarities = {}
    for file_name, vector_representation in article_embeddings.items():
        similarity = cosine_similarity([search_embedding], [vector_representation])[0][0]
        similarities[file_name] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    most_similar_articles = [article[0] for article in sorted_similarities[:5]]

    print("En uygun 5 makale:")
    for article, similarity in sorted_similarities[:5]:
        print(f"Makale Adı: {article}, Benzerlik Oranı: {similarity}")
        similarty_names.append(article)
        similarty_rates.append(similarity)
        
    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate in zip(similarty_names, similarty_rates)]    
    
    db.algorithm_for_sciBert.update_one(
        {'user_id': user_id},  
        {
            '$set': {
                'similarity_data': similarity_data,
                'tur': 'sciBert'
            }
        },
        upsert=True  
    )

def kisisellestirilmis_oneriler(request):
    user_id = request.session.get('_id')
    name = request.session.get('name')
    db_beğenilmisler = list(db.viewed.find({'user_id': user_id}, {'_id': 0, 'name': 1}))
    ozetler = []
    ozetler = set()

    for begenilmis_makale in db_beğenilmisler:
        makale_ismi = begenilmis_makale['name']
        
       
        makale_bilgisi = db.article_information.find_one({'filename': makale_ismi}, {'_id': 0,'summary': 1})
        
       
        if makale_bilgisi:
            ozet = makale_bilgisi['summary']
            ozetler.add(ozet)

   
    ozetler_duzenlenmis = [ozet.replace('\n', '').replace('\t', '') for ozet in ozetler]

    print(ozetler_duzenlenmis) 


    model_path = "cc.en.300.bin"
    word_list = ozetler_duzenlenmis
    average_vector_fastText = calculate_average_vector_for_fastText(word_list, model_path)
    personalised_for_fastText(average_vector_fastText,user_id)
    #print(average_vector_fastText)

    modelYoluu = "allenai/scibert_scivocab_uncased"
    word_listt = ozetler_duzenlenmis
    average_vector_sciBert = calculate_average_vector_for_SciBERT(word_listt, modelYoluu)
    personalised_for_sciBert(average_vector_sciBert, modelYoluu,user_id)
    #print(average_vector_sciBert)

    fasttext_articles_data = db.personalised_articles.find_one({'user_id': user_id, 'tur': 'fastText'})
    scibert_articles_data = db.personalised_articles.find_one({'user_id': user_id, 'tur': 'sciBert'})

    fasttext_article_names = [article['name'] for article in fasttext_articles_data['similarity_data']] if fasttext_articles_data else []
    scibert_article_names = [article['name'] for article in scibert_articles_data['similarity_data']] if scibert_articles_data else []

    # Makale detaylarını articles koleksiyonundan çek
    fasttext_articles = list(db.article_information.find({'filename': {'$in': fasttext_article_names}}))
    scibert_articles = list(db.article_information.find({'filename': {'$in': scibert_article_names}}))

    for article in fasttext_articles:
        article['mongo_id'] = article['_id'] 
    for article in scibert_articles:
        article['mongo_id'] = article['_id']

    return render(request, 'personalised_articles.html', {
        'name': name,
        'fasttext_articles': fasttext_articles,
        'scibert_articles': scibert_articles,
        })

def personalised_for_fastText(search_word_vector,user_id):
    
    file_names = []
    vector_representations = []
    similarty_names =[]
    similarty_rates =[]
        
    cursor = db.fastTextVectors.find()
    
    for document in cursor:
        file_names.append(document['file_name'])
        vector_representations.append(document['vector_representation'])

    
    vectors_np = np.array(vector_representations)


    search_word_vector_np = np.array([search_word_vector])


    similarity_scores = cosine_similarity(search_word_vector_np, vectors_np)


    top_matching_indices = np.argsort(similarity_scores[0])[-5:][::-1]


    print("\nEn çok eşleşen dosyalar:")
    for index in top_matching_indices:
        print(f"Dosya Adı: {file_names[index]}, Benzerlik Skoru: {similarity_scores[0][index]}")
        similarty_names.append(file_names[index])
        similarty_rates.append(similarity_scores[0][index])


    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate, in zip(similarty_names, similarty_rates)]
    
    db.personalised_articles.update_one(
        {'user_id': user_id,'tur': 'fastText'}, 
        {
            '$set': {
                'similarity_data': similarity_data,
                'tur': 'fastText'
            }
        },
        upsert=True  
    )  

def personalised_for_sciBert(average_vector_sciBert, model_name,user_id):
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    similarty_names =[]
    similarty_rates =[]
    search_embedding = average_vector_sciBert


    article_embeddings = {}
    for article in db.sciBertVectors.find():
        file_name = article['file_name']
        vector_representation = np.array(article['vector_representation'])
        article_embeddings[file_name] = vector_representation


    similarities = {}
    for file_name, vector_representation in article_embeddings.items():
        similarity = cosine_similarity([search_embedding], [vector_representation])[0][0]
        similarities[file_name] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    most_similar_articles = [article[0] for article in sorted_similarities[:5]]

    print("En uygun 5 makale:")
    for article, similarity in sorted_similarities[:5]:
        print(f"Makale Adı: {article}, Benzerlik Oranı: {similarity}")
        similarty_names.append(article)
        similarty_rates.append(similarity)
        
    similarity_data = [{'name': name, 'rate': rate, 'state':0} for name, rate in zip(similarty_names, similarty_rates)]    
    
    db.personalised_articles.update_one(
        {'user_id': user_id,'tur': 'sciBert'},  
        {
            '$set': {
                'similarity_data': similarity_data,
                'tur': 'sciBert'
            }
        },
        upsert=True  
    )

if not ilgiAlanlariBelirleme:
    folder_path = "keys"  

    all_words = read_files_in_folder(folder_path)


    word_counts = Counter(all_words)


    top_30_words = word_counts.most_common(30)

    for word, count in top_30_words:
        print(f"{word}: {count}")
        db.interests.insert_one({
                    'word': word
                })

if not nltkIslemleribool:
    # Burada NLTK kullanarak istenen NLP adımlarını gerçekleştirdim.
    nltkIslemleri(islenmemisInspec, islenmisInspec)

if not fastTextIslemleribool:
    # Burada NLTK kullanarak istenen NLP adımlarını gerçekleştirdim.
    modelYolu = "cc.en.300.bin"
    fastTextVectorKaydet(islenmisInspec, modelYolu)

if not sciBertIslemleribool:
    modelYolu = "allenai/scibert_scivocab_uncased"
    sciBertVectorKaydet(islenmisInspec, modelYolu)

if not process_text_filesBool:
    cleaned_dir = "inspec"
    keys_dir = "keys"
    process_text_files(cleaned_dir, keys_dir)
 