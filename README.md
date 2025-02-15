# **Emotion Detector** 🎭  
### **Real-Time Emotion Detection Using Deep Learning and OpenCV**  

A Python-based **real-time emotion detection system** that utilizes **Convolutional Neural Networks (CNNs)** to classify facial expressions. The model is trained on the **FER2013 dataset** and can detect **7 emotions**: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise**. It supports **real-time webcam detection** with probability percentages for each class.

---

## **🛠 Tools & Frameworks**
- **Python 3.8+**
- **PyTorch** – Deep learning framework for training and inference  
- **Torchvision** – Data transformations and dataset handling  
- **OpenCV** – Real-time face detection and image processing  
- **NumPy** – Handling numerical computations  
- **Matplotlib** – (Optional) Used for visualization  
- **Google Colab & Kaggle Notebook** – Used for training  
- **PyCharm** – Used for model deployment  

---

## **⚙️ Features**
✅ **Real-time emotion recognition** using OpenCV and a trained PyTorch model  
✅ **Live webcam detection** with emotion probabilities updated every second  
✅ **Pre-trained deep learning model** with CNN for facial expression recognition  
✅ **Fast and optimized face detection** using OpenCV's Haar cascade  
✅ **Structured modular codebase** for easy modifications and upgrades  

---

## **🛠️ How It Was Built**
### **1️⃣ Dataset Preparation**
- The **FER2013 dataset** was used, containing grayscale **48x48 facial images** classified into **7 emotions**.
- The dataset was structured into **train/test** folders with separate subfolders for each emotion.

### **2️⃣ Model Development**
- A **Convolutional Neural Network (CNN)** was designed using **PyTorch**.
- The model consists of **3 convolutional layers**, **ReLU activations**, **max pooling**, and **fully connected layers**.
- **Softmax activation** was used for classification.

### **3️⃣ Training & Evaluation**
- The model was trained using **CrossEntropyLoss** and **Adam optimizer**.
- It was trained in **Kaggle Notebook** and **Google Colab**.
- The trained model was saved as `"emotion_model.pth"` for later use.

### **4️⃣ Real-Time Emotion Detection**
- OpenCV was used to **capture frames** from the webcam.
- The **Haar cascade classifier** was used for **face detection**.
- Detected faces were preprocessed and passed through the **CNN model**.
- Emotion probabilities were displayed **live on the webcam feed**.

### **5️⃣ Optimization & Deployment**
- The script was **optimized for performance**, updating **predictions every 1 second**.
- The project was modularized for **scalability**.
- Future plans include deploying as a **web or mobile app**.

---

## **💡 Future Ideas**
- **Enhanced Emotion Analysis:** Improve accuracy with a more diverse dataset.
- **Comprehensive Analytics:** Track detected emotions over time for deeper insights.
- **Mobile App Implementation:** Expand functionality to Android and iOS.

## **🪄 Future Improvements**
✅ **Adding real-time emotion graphs** for better visualization.  
✅ **Integration with AI assistants** to respond based on detected emotions.  
✅ **Deploying the system as a web service** using Flask or FastAPI.  
✅ **Optimizing model performance** to run faster on low-end devices.

---

## **🎥 Demo Video**
📌 *Coming Soon!* A demonstration video showcasing real-time emotion detection.

---

## **📜 License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **📧 Contact**
For questions, suggestions, or support, feel free to reach out:

- **Name:** Rayan  
- **GitHub:** [R2RyAn](https://github.com/R2RyAn)  
- **Email:** [rayandajani21@gmail.com](mailto:rayandajani21@gmail.com)  

🚀 **Happy Coding!** 🎭🔥
