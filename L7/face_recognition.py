import numpy as np

from L5.knn import KNN


class FaceRecognition:
    def __init__(self, k_neighbours=1):
        self.training_data = None
        self.images_per_person_teaching = 5
        self.training_targets = None
        self.training_targets_faces = None
        self.face_id_knn = KNN(k=k_neighbours)
        self.testing_data = None
        self.images_per_person_testing = 2
        self.predicted_face_ids = None

    def fit(self, training_data, *, images_per_person_teaching=5):
        self.training_data = training_data
        self.images_per_person_teaching = images_per_person_teaching
        self.training_targets = np.array(range(self.training_data.shape[0]))
        self.face_id_knn.fit(self.training_data, self.training_targets)

    def predict(self, testing_data, *, images_per_person_testing=2):
        self.testing_data = testing_data
        self.images_per_person_testing = images_per_person_testing
        prediction = self.face_id_knn.predict(self.testing_data)
        return prediction // self.images_per_person_teaching

    @staticmethod
    def person_id(image_idx, images_per_person=5):
        return image_idx // images_per_person

    def _prepare_results(self):
        self.predicted_face_ids = self.face_id_knn.predict(self.testing_data)
        self.euclidean_distances = self.face_id_knn.euclidean_distance(self.training_data, self.testing_data)

    def print_results(self, test_data):
        self.testing_data = test_data
        self._prepare_results()
        print("pred. id", "eucl. distance", "pr. pers.", "pers.", "correct", "test id.", sep="\t")
        for i, (predicted_face_id, euclidean_distances_from_predicted
                ) in enumerate(zip(self.predicted_face_ids, self.euclidean_distances)):
            print(predicted_face_id, int(euclidean_distances_from_predicted[i]),
                  FaceRecognition.person_id(predicted_face_id), FaceRecognition.person_id(i, images_per_person=2),
                  FaceRecognition.person_id(predicted_face_id) == FaceRecognition.person_id(i, images_per_person=2), i,
                  sep="\t")

    def score(self, test_data):
        self.testing_data = test_data
        testing_targets_faces = np.array(
            [self.images_per_person_testing * [i]
             for i in range(self.testing_data.shape[0] // self.images_per_person_testing)]).flatten()

        predicted_labels = self.predict(self.testing_data,
                                        images_per_person_testing=self.images_per_person_testing)
        return sum(testing_targets_faces == predicted_labels) / len(testing_targets_faces)


if __name__ == "__main__":
    import scipy.io
    from sklearn import decomposition

    training_images = scipy.io.loadmat('data/ReducedImagesForTraining.mat')["images"].T
    testing_images = scipy.io.loadmat('data/ReducedImagesForTesting.mat')["images"].T
    face_recognition = FaceRecognition()
    face_recognition.fit(training_images)
    predicted_faces = face_recognition.predict(testing_images)
    print(predicted_faces)
    face_recognition.print_results(testing_images)
    print(face_recognition.score(testing_images))

    pca = decomposition.PCA(n_components=37)
    pca.fit(training_images)
    training_images_pca = pca.transform(training_images)
    testing_images_pca = pca.transform(testing_images)
    print(training_images_pca.shape, testing_images_pca.shape)

    face_recognition = FaceRecognition()
    face_recognition.fit(training_images_pca)
    print(face_recognition.score(testing_images_pca))
