import faiss
from keras import Model
import numpy as np
from keras._tf_keras.keras.preprocessing import image
import os
from keras.src.applications.resnet import preprocess_input, ResNet50
import pickle


class SimilarityServiceV3:
    # Base path setup
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f"{base_path}/dataset2/train"
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
    def get_embeddings(self, img_dirs):
        embeddings = []
        image_paths = []

        for img_dir in img_dirs:
            dir_path = os.path.join(self.dataset_path, img_dir)
            files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                     if f.endswith(('jpg', 'jpeg', 'png', 'gif'))]
            for path in files:
                img = image.load_img(path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                embedding = self.model.predict(img_array)[0]
                embeddings.append(embedding)
                image_paths.append(self.extract_image_relative_path(path, self.dataset_path))

        return np.array(embeddings).astype('float32'), image_paths

    def build_faiss_index(self):
        embeddings, image_paths = self.get_embeddings(self.img_dirs)
        dim = embeddings.shape[1]

        # Using IndexFlatL2 for cosine-like behavior (normalize first)
        index = faiss.IndexFlatL2(dim)

        # Normalize embeddings (for cosine similarity)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # Save for later use
        faiss.write_index(index, f"{self.base_path}/faiss.index")
        pickle.dump(image_paths, open(f"{self.base_path}/image_paths.pkl", "wb"))

        return index, image_paths

    def load_faiss_index(self):
        index = faiss.read_index(f"{self.base_path}/faiss.index")
        image_paths = pickle.load(open(f"{self.base_path}/image_paths.pkl", "rb"))
        return index, image_paths

    def recompute_embedding(self,img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        embedding = self.model.predict(img_array)
        return embedding

    def find_k_similar_images(self, target, k=5):
        index_path = f"{self.base_path}/faiss.index"
        if not os.path.exists(index_path):
            print("FAISS index not found. Building one...")
            self.build_faiss_index()
        index, image_paths = self.load_faiss_index()

        # Preprocess query image
        query_embedding = self.recompute_embedding(target)[0].astype('float32')
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        # Search
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        results = [(1 - distances[0][i], image_paths[indices[0][i]]) for i in range(k)]  # Convert L2 to similarity

        return results

    import os

    def evaluate_similarity_model(self, test_base_path, k=1):
        correct = 0
        total = 0
        test_images = []

        # Build a list of (image_path, true_label)
        for root, dirs, files in os.walk(test_base_path):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(root, file)
                    # Extract label from folder name (e.g., /test/Bean/img1.jpg → 'Bean')
                    true_label = os.path.basename(os.path.dirname(img_path))
                    test_images.append((img_path, true_label))

        print(f"Evaluating {len(test_images)} test images...")

        for img_path, true_label in test_images:
            results = self.find_k_similar_images(img_path, k=k)

            # Check if the predicted similar images match the true label
            matches = [true_label in os.path.normpath(path).split(os.sep) for score, path in results]

            correct += sum(matches)
            total += len(matches)

        accuracy = correct / total if total != 0 else 0
        print(f"Top-{k} accuracy: {accuracy:.2%}")


base_path = os.path.dirname(os.path.abspath(__file__))
test_dataset_path = f"{base_path}/dataset2/test"

model = SimilarityServiceV3()
model.evaluate_similarity_model(test_dataset_path)

