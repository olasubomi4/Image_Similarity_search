from keras import Model
import numpy as np
from keras._tf_keras.keras.preprocessing import image
import os
from keras.src.applications.resnet import preprocess_input, ResNet50
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class SimilarityService:
    # Base path setup
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f"{base_path}/datasets/"
    loadExistingEmbedding = True

    # Load ResNet50 base model
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Categories
    img_dirs = [
        "Bean", "Bitter_Gourd", "Tomato", "Brinjal", "Broccoli",
        "Cabbage", "Carrot", "Cucumber", "Papaya", "Potato", "Pumpkin"
    ]
    # Utility to extract relative image path
    def extract_image_relative_path(self,text, text_be_removed):
        return ' '.join(text.replace(text_be_removed, "").split())

    # Batched version of get_embeddings
    def get_embeddings(self,img_dirs):
        global dataset_path
        embeddings_list = []
        image_paths = []

        # Gather all image paths
        for img_dir in img_dirs:
            dir_path = os.path.join(dataset_path, img_dir)
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                     if f.endswith(('jpg', 'jpeg', 'png', 'gif'))]
            image_paths.extend(files)

        # Load all images and preprocess
        img_arrays = []
        rel_paths = []

        for path in image_paths:
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_arrays.append(img_array)
            rel_paths.append(self.extract_image_relative_path(path, dataset_path))

        # Convert to numpy batch
        img_batch = np.array(img_arrays)
        embeddings = self.model.predict(img_batch)

        # Store (embedding, path) pairs
        for emb, path in zip(embeddings, rel_paths):
            embeddings_list.append((emb, path))

        return embeddings_list

    # Embedding for one query image
    def recompute_embedding(self,img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        embedding = self.model.predict(img_array)
        return embedding

    # Load previously saved embeddings
    def retrieve_existing_embeddings(self,filepath):
        return pickle.load(open(filepath, 'rb'))



    def get_embedding(self):
    # Load or generate embeddings
        if self.loadExistingEmbedding:
            self.embeddings_list = self.retrieve_existing_embeddings(self.base_path + "/dic.pkl")
        else:
            self.embeddings_list = self.recompute_embedding(self.img_dirs)
            pickle.dump(self.embeddings_list, open(self.base_path + "/dic.pkl", 'wb'))

        return self.embeddings_list

    # Find top K similar images using cosine similarity
    def find_k_similar_images(self,target, k=5):
        self.embeddings_list=self.get_embedding()
        scores = []
        target_embedding = self.recompute_embedding(target)
        for emb, path in self.embeddings_list:
            emb = np.expand_dims(emb, axis=0)
            score = cosine_similarity(target_embedding, emb)[0][0]
            scores.append((score, path))
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:k]

    # # Find and display top matches
    # top_results = find_k_similar_images(target, k=5)
    #
    # print("\nTop 5 similar images:")
    # for i, (score, path) in enumerate(top_results, start=1):
    #     print(f"{i}. {path} with similarity score: {score}")
    #
    # # Visualize the results
    # fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    # for i, (_, img_rel_path) in enumerate(top_results):
    #     img = Image.open(os.path.join(dataset_path, img_rel_path))
    #     axes[i].imshow(img)
    #     axes[i].axis('off')
    #
    # plt.tight_layout()
    # plt.show()
