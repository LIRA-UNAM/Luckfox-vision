from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QMessageBox, QGraphicsDropShadowEffect
from PyQt5.QtGui import QFont, QPalette, QBrush, QLinearGradient, QColor, QCursor
from PyQt5.QtCore import Qt
import sys, subprocess, os

class YOLOTrainer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MODEL CONVERTER")
        self.setGeometry(100, 100, 750, 550)  # ventana grande

        # Fondo azul marino
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(10, 25, 47))  # Azul marino
        self.setPalette(palette)

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(20)

        # Título
        title = QLabel("MODEL HELPER")
        title.setFont(QFont("Helvetica", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        title.setStyleSheet("color: #5dade2;")  # Azul claro

        neon_effect = QGraphicsDropShadowEffect()
        neon_effect.setColor(QColor(93, 173, 226))  # Azul brillante
        neon_effect.setBlurRadius(40)  # difuminado para efecto neón
        neon_effect.setOffset(0, 0)    # sin sombra, solo glow
        title.setGraphicsEffect(neon_effect)

        main_layout.addWidget(title)

        # Dataset folder
        folder_layout = QHBoxLayout()

        self.path_input = QLineEdit()
        self.path_input.setFixedWidth(480)
        self.path_input.setPlaceholderText("Select Dataset directory (Images/ + JSONs/)")
        self.path_input.setStyleSheet("background-color: white; border-radius: 5px; padding: 5px;")
        folder_layout.addWidget(self.path_input)

        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setStyleSheet(self.button_style("#2980b9"))
        self.btn_browse.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_browse.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.btn_browse)
        main_layout.addLayout(folder_layout)
        self.btn_browse.setFixedWidth(120)   # ancho
        self.btn_browse.setFixedHeight(40)   # alto
        
        # --- Select YOLOv5 directory ---
        yolo_layout = QHBoxLayout()

        self.yolo_path_input = QLineEdit()
        self.yolo_path_input.setFixedWidth(480)
        self.yolo_path_input.setPlaceholderText("Select Yolo Directory ")
        self.yolo_path_input.setStyleSheet("background-color: white; border-radius: 5px; padding: 5px;")
        yolo_layout.addWidget(self.yolo_path_input)

        self.btn_yolo_browse = QPushButton("Browse")
        self.btn_yolo_browse.setStyleSheet(self.button_style("#2980b9"))
        self.btn_yolo_browse.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_yolo_browse.clicked.connect(self.select_yolo_folder)
        yolo_layout.addWidget(self.btn_yolo_browse)
        self.btn_yolo_browse.setFixedWidth(120)   # ancho
        self.btn_yolo_browse.setFixedHeight(40)   # alto
        main_layout.addLayout(yolo_layout)

        # Image size and epochs
        params_layout = QHBoxLayout()
        label_font = QFont("Helvetica", 12, QFont.Bold)
        input_font = QFont("Helvetica", 12)

        self.img_input = QLineEdit("640")
        self.img_input.setFixedWidth(60)
        self.img_input.setFont(input_font)
        self.img_input.setStyleSheet("background-color: white; border-radius: 5px; padding: 5px;")

        self.epochs_input = QLineEdit("50")
        self.epochs_input.setFixedWidth(70)
        self.epochs_input.setFont(input_font)
        self.epochs_input.setStyleSheet("background-color: white; border-radius: 5px; padding: 5px;")

        img_label = QLabel("Image size:")
        img_label.setFont(label_font)
        img_label.setStyleSheet("color: white;")
        epochs_label = QLabel("Epochs:")
        epochs_label.setStyleSheet("color: white;")
        epochs_label.setFont(label_font)

        params_layout.addStretch()
        params_layout.addWidget(img_label)
        params_layout.addWidget(self.img_input)
        params_layout.addSpacing(40)
        params_layout.addWidget(epochs_label)
        params_layout.addWidget(self.epochs_input)
        params_layout.addStretch()
        main_layout.addLayout(params_layout)

        # Buttons in 2 columns
        buttons = [
            ("Prepare dataset", self.prepare_dataset, "#2980b9"),
            ("Train YOLOv5", self.train_yolov5, "#2980b9"),
            ("Export to ONNX", self.export_onnx, "#2980b9"),
            ("Export to RKNN", self.export_rknn, "#2980b9"),
        ]

        for i in range(0, len(buttons), 2):
            row_layout = QHBoxLayout()
            row_layout.addStretch()
            for j in range(2):
                if i + j < len(buttons):
                    text, func, color = buttons[i + j]
                    btn = QPushButton(text)
                    btn.setStyleSheet(self.button_style(color))
                    btn.setCursor(QCursor(Qt.PointingHandCursor))
                    btn.setFixedWidth(200)
                    btn.setFixedHeight(50)
                    btn.clicked.connect(func)
                    row_layout.addWidget(btn)
                    row_layout.addSpacing(20)
            row_layout.addStretch()
            main_layout.addLayout(row_layout)

        self.setLayout(main_layout)

    # -------------------- Styles --------------------
    def button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-weight: bold;
                border-radius: 8px;
                border: 2px solid #5dade2; /* azul claro */
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 30);
                color: #5dade2;
                border: 2px solid white;
            }}
        """
    # -------------------- Functions --------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if folder:
            self.path_input.setText(folder) #path to work

    def select_yolo_folder (self):
        folder = QFileDialog.getExistingDirectory(self,"Select Yolo Directory")
        if folder :
            self.yolo_path_input.setText(folder)

    def prepare_dataset(self):
        folder = self.path_input.text()
        if not os.path.exists(folder):
            QMessageBox.critical(self, "Error", "Invalid folder")
            return

        # ---- Check pair images and json ----
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        jsons = [f for f in os.listdir(folder) if f.lower().endswith('.json')]

        image_names = set(os.path.splitext(f)[0] for f in images)
        json_names = set(os.path.splitext(f)[0] for f in jsons)

        images_without_json = image_names - json_names
        json_without_image = json_names - image_names

        msg = ""
        if images_without_json:
            msg += f"{len(images_without_json)} images without .json\n"
        if json_without_image:
            msg += f"{len(json_without_image)} .json files without image\n"

        if msg:
            msg += "Do you want to delete these files?"
            reply = QMessageBox.question(self, "Orphan files detected", msg,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # delete images without .json
                for name in images_without_json:
                    path = os.path.join(folder, name + os.path.splitext(next(f for f in images if os.path.splitext(f)[0] == name))[1])
                    os.remove(path)
                # delete .json without image
                for name in json_without_image:
                    path = os.path.join(folder, name + '.json')
                    os.remove(path)

        # ---- Run labelme2yolo ----
        try:
            subprocess.run(["labelme2yolo", "--json_dir", folder, "--val_size", "0.2"], check=True)
            QMessageBox.information(self, "Success", "Dataset prepared successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    def train_yolov5(self):
        folder_yolo = self.yolo_path_input.text()
        folder = self.path_input.text()
        yaml_path = os.path.join(folder, "YOLODataset", "dataset.yaml")
        yolo_path = os.path.join(folder_yolo,"train.py")
        onnx_path = os.path.join(folder_yolo,"export.py")
        best_path = os.path.join(folder_yolo,"runs/train/exp/weights/best.pt")
        if not os.path.exists(yaml_path):
            QMessageBox.critical(self,"Error",f"dataset.yaml not found in {yaml_path}")
            return

        try:
            subprocess.run([
                "python3", yolo_path,
                "--img", self.img_input.text(),
                "--batch", "16",
                "--epochs", self.epochs_input.text(),
                "--data", yaml_path,
                "--weights", "yolov5s.pt"
            ], check=True)
            QMessageBox.information(self, "Training", "Training completed!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def export_onnx(self):
        try:
            subprocess.run(["python3", onnx_path, "--weights", best_path, "--include", "onnx"], check=True)
            QMessageBox.information(self, "Export", "Exported to ONNX!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def export_rknn(self):
        try:
            subprocess.run(["python3", "rknn_toolkit/rknn_convert.py", "--onnx", "runs/train/exp/weights/best.onnx"], check=True)
            QMessageBox.information(self, "Export", "Exported to RKNN!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

# -------------------- Run app --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = YOLOTrainer()
    window.show()
    sys.exit(app.exec_())
