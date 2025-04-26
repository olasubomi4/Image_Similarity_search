# from keras import Model
# import numpy as np
# from keras._tf_keras.keras.preprocessing import image
# import os
# from keras.src.applications.resnet import preprocess_input, ResNet50
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle
# import matplotlib.pyplot as plt
# from PIL import Image
#
# base_path = os.path.dirname(os.path.abspath(__file__))
# dataset_path=f"{base_path}/datasets/"
# base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
# model = Model(inputs=base_model.input, outputs=base_model.output)
# loadExistingEmbedding=False;
#
# def extract_image_relative_path(text,text_be_removed):
#     return ' '.join(text.replace(text_be_removed, "").split())
#
# def get_embeddings(img_dirs):
#     di=dict();
#     for img_dir in img_dirs:
#         img_dir=dataset_path+img_dir
#         image_files=[os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('jpg', 'png', 'jpeg', 'gif'))]
#
#         for img_path in image_files:
#             img = image.load_img(img_path, target_size=(224, 224))
#             img_array = image.img_to_array(img)
#             img_array = np.expand_dims(img_array, axis=0)
#             img_array = preprocess_input(img_array)
#             embedding = model.predict(img_array)
#             embedding_tuple= tuple(embedding.flatten())
#             di[embedding_tuple] = extract_image_relative_path(img_path,dataset_path);
#     return di
#
# img_dirs=["Bean","Bitter_Gourd","Tomato","Brinjal","Broccoli","Cabbage","Carrot","Cucumber"
#     ,"Papaya","Potato","Pumpkin"]
#
# def get_embedding(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     embedding = model.predict(img_array)
#     return embedding
#
# def retrieveExistingEmbeddings(embeddings):
#     dic=pickle.load(open(embeddings, 'rb'))
#     return dic
#
#
# if loadExistingEmbedding:
#     dic=retrieveExistingEmbeddings(base_path+"/dic.pkl")
# else:
#     dic=get_embeddings(img_dirs)
#
# target=get_embedding(dataset_path+"Tomato/0001.jpg")
#
#
# def findKSimilarImages(target,dic):
#     similarity_scores=[]
#     for embedding,img_path in dic.items():
#         embedding = np.array(embedding).reshape(1,-1)
#         score = cosine_similarity(target, embedding)
#         similarity_scores.append((score,img_path))
#     similarity_scores.sort(reverse=True,key=lambda x:x[0])
#     return similarity_scores
#
#
# pickle.dump(dic,open('dic.pkl','wb'))
# result=findKSimilarImages(target,dic);
#
# print(f"{result[0][1]} is the most similar to target with score result {result[0][0]}")
# print(f"{result[1][1]} is the second most similar to target with score result {result[1][0]}")
# print(f"{result[2][1]} is the third most similar to target with score result {result[2][0]}")
# print(f"{result[3][1]} is the fourth most similar to target with score result {result[3][0]}")
# print(f"{result[4][1]} is the Fifth most similar to target with score result {result[4][0]}")
#
#
# fig, axes = plt.subplots(1, 5, figsize=(15, 5))
#
# for i in range(5):
#     img = Image.open(dataset_path+result[i][1])
#     axes[i].imshow(img)
#     axes[i].axis('off')
#
# plt.show()
#
#
#
