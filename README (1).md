# 🧬 Blood Group Detection Using Fingerprint with Image Processing

This project presents an innovative, non-invasive approach to determining human blood groups using fingerprint images and deep learning. By integrating computer vision, artificial intelligence, and a web-based interface, it enables fast and accurate predictions suitable for emergency medical settings and low-resource areas.

---

## 📘 Project Summary

- **Title:** Blood Group Detection Using Fingerprint with Image Processing
- **Institute:** Jain College of Engineering and Technology, Hubballi
- **Year:** 2024 – 2025
- **Submitted by:** Abhinandan Gaikwad–2JH21CS002, Anjum Nadaf–2JH21CS017, Bhramarambika Joshi–2JH21CS028, Sooraj Emmi–2JH21CS100
- **Guide:** Prof. Shilpa Hadimani

---

## ✅ Features

- Predicts ABO and Rh blood groups from fingerprint images
- Uses CNN trained on fingerprint patterns
- Web application using Flask for user interface
- GUI prediction using Tkinter
- User login, signup, and profile management with SQLite
- Result page with predicted blood group
- Admin and user dashboards

---

## 🧪 Tech Stack

| Layer       | Tools & Frameworks               |
| ----------- | -------------------------------- |
| Frontend    | HTML, CSS, JavaScript, Bootstrap |
| Backend     | Python (Flask, Tkinter)          |
| ML / DL     | PyTorch, torchvision, PIL        |
| Image Proc. | OpenCV, scikit-image             |
| Data Mgmt   | SQLite3                          |
| GUI         | Tkinter                          |
| Dev Tools   | Jupyter Notebook, VS Code        |

---

## 📂 Project Structure

```
├── app.py                        # Main Flask web app
├── fingerprint_blood_group_model.pkl  # Trained PyTorch model
├── templates/                   # HTML templates
├── static/uploads/              # Uploaded fingerprint images
├── dataset_blood_group/        # Training images organized by blood group
├── notebook.ipynb              # Jupyter notebook for training & GUI testing
├── users.db                    # SQLite database for user data
└── README.md
```

---

## 🧠 Model: `SimpleCNN`

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 8)  # For 8 blood groups

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

---

## 📓 Jupyter Workflow Highlights

- Loads dataset using `ImageFolder`
- Applies resizing, normalization
- Trains CNN for 25 epochs with SGD optimizer
- Uses `Tkinter` to test predictions via file upload

---

## 🌐 Flask Application

- `/predict`: Upload fingerprint and get blood group
- `/signup`, `/login`, `/logout`: User authentication
- `/profile`: Update user info
- `/about`, `/team`, `/Accuracy`: Informational pages
- Prediction view: `login2.html`, Result view: `result.html`

---

## ⚙️ Required pip installations

Create a separate environment in Anaconda Navigator and then install all the below mentioned dependices....

pip install tensorflow
conda install -c conda-forge keras
conda install -c anaconda scikit-learn
conda install -c conda-forge opencv
conda install -c anaconda scikit-image
conda install -c anaconda flask
pip install pandas
pip install werkzeug==2.3.7
pip install torch==2.0.1
pip install torchvision==0.15.2
pip install flask-wtf
pip install WTForms==2.3.3

---

## 🚀 Future Scope

- Use multi-biometric features (e.g. fingerprint + iris)
- Expand training dataset for better generalization
- Integrate model into national ID and mobile apps
- Real-time prediction kits for emergency care

---
