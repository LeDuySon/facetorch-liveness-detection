import numpy as np
import cv2
import requests

from PIL import Image

class HistFAS:
    """Ref: https://github.com/ee09115/spoofing_detection"""
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.api_uri = "http://fas:5000/predict"
        
    @staticmethod
    def calc_hist(img):
        histogram = [0] * 3
        for j in range(3):
            histr = cv2.calcHist([img], [j], None, [256], [0, 256])
            histr *= 255.0 / histr.max()
            histogram[j] = histr
        return np.array(histogram)
    
    def get_embedding(self, face_img):
        img_ycrcb = cv2.cvtColor(face_img.copy(), cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(face_img.copy(), cv2.COLOR_BGR2LUV)
        
        ycrcb_hist = self.calc_hist(img_ycrcb)
        luv_hist = self.calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))
        
        return feature_vector
    
    def make_request(self, data, attack_type):
        # Define the API endpoint URL
        url = f"{self.api_uri}/{attack_type}"

        # Prepare the request payload with the data as a NumPy array
        payload = {"data": data.tolist()}

        try:
            # Make a POST request to the API endpoint
            response = requests.post(url, json=payload)

            # Check the response status code
            if response.status_code == 200:
                # Get the prediction from the response JSON
                prediction = response.json()["prediction"]
                return prediction
            else:
                print("Error:", response.json()["error"])

        except requests.exceptions.RequestException as e:
            print("Error:", e)
    
    def check_print_attack(self, feature):
        attack_type = "print" 
        prediction = self.make_request(feature, attack_type)
        
        return prediction[0][1]
    
    def check_replay_attack(self, feature):
        attack_type = "replay" 
        prediction = self.make_request(feature, attack_type)
        
        return prediction[0][1]
    
    def check(self, face_img):
        face_img = np.array(face_img)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        
        feature_embed = self.get_embedding(face_img) 
        
        print_attack_prob = self.check_print_attack(feature_embed.copy())
        replay_attack_prob = self.check_replay_attack(feature_embed) 
        
        print("print attack prob: ", print_attack_prob)
        print("replay attack prob: ", replay_attack_prob)
        
        if(print_attack_prob >= self.threshold):
            return False 
        
        return True
        
    