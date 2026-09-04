<<<<<<< ours
import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QSlider, QLabel, QMessageBox, QPushButton)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

class MITKViewer(QMainWindow):
    def __init__(self, vol_mitk, mask_mitk=None):
        super().__init__()
        self.setWindowTitle("Visualiseur Médical - Multi-Vues (Volume + Seg)")
        self.setStyleSheet("background-color: #121212; color: white;")
        
        self.volume = vol_mitk 
        self.mask = mask_mitk # Stockage du masque (déjà transposé)

        # Création d'une colormap : 0 = transparent, 1 = Rouge vif
        # On utilise une liste de couleurs pour ListedColormap
        self.cmap_mask = mcolors.ListedColormap([(0,0,0,0), (1,0,0,1)]) # RGBA

        # --- LAYOUT PRINCIPAL ---
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- SIDEBAR ---
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(15, 20, 15, 20)
        
        title_nav = QLabel("NAVIGATION")
        title_nav.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #AAAAAA;")
        sidebar.addWidget(title_nav)
        
        # Initialisation basée sur les dimensions réelles du volume
        self.slider_ax, self.lbl_ax = self.create_mitk_slider("AXIAL", self.volume.shape[0], sidebar, "#FF5555")
        self.slider_cor, self.lbl_cor = self.create_mitk_slider("CORONAL", self.volume.shape[1], sidebar, "#5555FF")
        self.slider_sag, self.lbl_sag = self.create_mitk_slider("SAGITTAL", self.volume.shape[2], sidebar, "#55FF55")
        
        sidebar.addSpacing(20)

        self.btn_reset = QPushButton("RESET VIEWS")
        self.btn_reset.setMinimumHeight(45)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #00E5FF; color: #000000; font-weight: bold; 
                border-radius: 8px; border: 2px solid #00B8D4;
            }
            QPushButton:hover { background-color: #64FFDA; }
        """)
        self.btn_reset.clicked.connect(self.reset_views)
        sidebar.addWidget(self.btn_reset)
        
        sidebar.addStretch()
        main_layout.addLayout(sidebar, 1)

        # --- GRILLE DES VUES ---
        grid_layout = QVBoxLayout()
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        
        self.canvas_ax, self.ax_ax = self.create_view_canvas("axial")
        self.canvas_sag, self.ax_sag = self.create_view_canvas("sagittal")
        self.canvas_cor, self.ax_cor = self.create_view_canvas("coronal")
        self.canvas_empty, self.ax_empty = self.create_view_canvas("3d")
        
        row1.addWidget(self.canvas_ax); row1.addWidget(self.canvas_sag)
        row2.addWidget(self.canvas_cor); row2.addWidget(self.canvas_empty)
        
        grid_layout.addLayout(row1); grid_layout.addLayout(row2)
        main_layout.addLayout(grid_layout, 5)

        self.update_plots()
        self.showMaximized()

    def create_mitk_slider(self, name, max_val, layout, color):
        lbl = QLabel(f"{name} (Slice 1 / {max_val})")
        lbl.setStyleSheet(f"color: {color}; font-weight: bold; margin-top: 15px;")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, max_val - 1)
        slider.setValue(max_val // 2)
        slider.valueChanged.connect(self.update_plots)
        layout.addWidget(lbl)
        layout.addWidget(slider)
        return slider, lbl

    def create_view_canvas(self, view_type):
        fig = Figure(facecolor='black')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        canvas.wheelEvent = lambda event: self.handle_scroll(event, view_type)
        return canvas, ax

    def handle_scroll(self, event, view_type):
        step = 1 if event.angleDelta().y() > 0 else -1
        if view_type == "axial": self.slider_ax.setValue(self.slider_ax.value() + step)
        elif view_type == "coronal": self.slider_cor.setValue(self.slider_cor.value() + step)
        elif view_type == "sagittal": self.slider_sag.setValue(self.slider_sag.value() + step)
        event.accept()

    def reset_views(self):
        self.slider_ax.setValue(self.volume.shape[0] // 2)
        self.slider_cor.setValue(self.volume.shape[1] // 2)
        self.slider_sag.setValue(self.volume.shape[2] // 2)

    def update_plots(self):
        z, y, x = self.slider_ax.value(), self.slider_cor.value(), self.slider_sag.value()
        
        # Sécurité pour ne pas sortir des dimensions après resize potentiel
        z = min(z, self.volume.shape[0]-1)
        y = min(y, self.volume.shape[1]-1)
        x = min(x, self.volume.shape[2]-1)

        self.lbl_ax.setText(f"AXIAL (Slice {z+1} / {self.volume.shape[0]})")
        self.lbl_cor.setText(f"CORONAL (Slice {y+1} / {self.volume.shape[1]})")
        self.lbl_sag.setText(f"SAGITTAL (Slice {x+1} / {self.volume.shape[2]})")

        def draw_image(ax, canvas, data, mask_slice, title, color, lh, lv, ch, cv):
            ax.clear()
            # Affichage du volume
            ax.imshow(data, cmap='gray', aspect='equal', interpolation='nearest')
            
            # Affichage du masque par-dessus
            if mask_slice is not None and mask_slice.shape == data.shape:
                # On s'assure que le masque est binaire (0 ou 1)
                m_bin = (mask_slice > 0).astype(np.uint8)
                if np.any(m_bin):
                    ax.imshow(m_bin, cmap=self.cmap_mask, alpha=0.5, aspect='equal', interpolation='nearest', zorder=10)
            
            ax.axhline(lh, color=ch, linewidth=0.8, alpha=0.6)
            ax.axvline(lv, color=cv, linewidth=0.8, alpha=0.6)
            ax.text(0.02, 0.98, title, transform=ax.transAxes, color=color, fontweight='bold', va='top')
            ax.axis('off')
            canvas.draw()

        # Tranches Volume et Masque
        draw_image(self.ax_ax, self.canvas_ax, self.volume[z, :, :], 
                   self.mask[z, :, :] if self.mask is not None else None, 
                   "AXIAL", "#FF5555", y, x, "blue", "green")
        
        draw_image(self.ax_sag, self.canvas_sag, self.volume[:, :, x], 
                   self.mask[:, :, x] if self.mask is not None else None, 
                   "SAGITTAL", "#55FF55", z, y, "red", "blue")
        
        draw_image(self.ax_cor, self.canvas_cor, self.volume[:, y, :], 
                   self.mask[:, y, :] if self.mask is not None else None, 
                   "CORONAL", "#5555FF", z, x, "red", "green")
        
        self.ax_empty.axis('off')
        self.canvas_empty.draw()

def valider_recalage(volume_data, mask_data=None):
    # --- GESTION DES DIMENSIONS ---
    if mask_data is not None and volume_data.shape != mask_data.shape:
        print(f"Mismatch: Volume {volume_data.shape} vs Mask {mask_data.shape}. Adaptation...")
        new_mask = np.zeros(volume_data.shape, dtype=mask_data.dtype)
        z_c, y_c, x_c = [min(v, m) for v, m in zip(volume_data.shape, mask_data.shape)]
        new_mask[:z_c, :y_c, :x_c] = mask_data[:z_c, :y_c, :x_c]
        mask_data = new_mask

    # --- TRANSPOSITION ET NORMALISATION ---
    vol_mitk = np.transpose(volume_data, (0, 2, 1))
    max_v = np.max(vol_mitk) if np.max(vol_mitk) > 0 else 1.0
    data_ui = (vol_mitk / max_v * 255).astype(np.uint8)
    
    mask_ui = None
    if mask_data is not None:
        mask_ui = np.transpose(mask_data, (0, 2, 1))

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = MITKViewer(data_ui, mask_ui)
    viewer.show()

    while viewer.isVisible():
        app.processEvents()

    if QMessageBox.question(None, "Decision", "Passer à l'étape suivante ?", 
                            QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
        sys.exit()

if __name__ == "__main__":
    # Test avec un volume 320x320 et un masque 388 (pour simuler votre erreur)
    data_test = np.random.randint(0, 100, (320, 256, 256))
    mask_test = np.zeros((388, 256, 256)) # Plus grand que le volume
    mask_test[100:200, 100:200, 100:200] = 1 
    
    valider_recalage(data_test, mask_test)
=======
import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QSlider, QLabel, QMessageBox, QPushButton)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

class MITKViewer(QMainWindow):
    def __init__(self, vol_mitk, mask_mitk=None):
        super().__init__()
        self.setWindowTitle("Visualiseur Médical - Multi-Vues (Volume + Seg)")
        self.setStyleSheet("background-color: #121212; color: white;")
        
        self.volume = vol_mitk 
        self.mask = mask_mitk # Stockage du masque (déjà transposé)

        # Création d'une colormap : 0 = transparent, 1 = Rouge vif
        # On utilise une liste de couleurs pour ListedColormap
        self.cmap_mask = mcolors.ListedColormap([(0,0,0,0), (1,0,0,1)]) # RGBA

        # --- LAYOUT PRINCIPAL ---
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- SIDEBAR ---
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(15, 20, 15, 20)
        
        title_nav = QLabel("NAVIGATION")
        title_nav.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #AAAAAA;")
        sidebar.addWidget(title_nav)
        
        # Initialisation basée sur les dimensions réelles du volume
        self.slider_ax, self.lbl_ax = self.create_mitk_slider("AXIAL", self.volume.shape[0], sidebar, "#FF5555")
        self.slider_cor, self.lbl_cor = self.create_mitk_slider("CORONAL", self.volume.shape[1], sidebar, "#5555FF")
        self.slider_sag, self.lbl_sag = self.create_mitk_slider("SAGITTAL", self.volume.shape[2], sidebar, "#55FF55")
        
        sidebar.addSpacing(20)

        self.btn_reset = QPushButton("RESET VIEWS")
        self.btn_reset.setMinimumHeight(45)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #00E5FF; color: #000000; font-weight: bold; 
                border-radius: 8px; border: 2px solid #00B8D4;
            }
            QPushButton:hover { background-color: #64FFDA; }
        """)
        self.btn_reset.clicked.connect(self.reset_views)
        sidebar.addWidget(self.btn_reset)
        
        sidebar.addStretch()
        main_layout.addLayout(sidebar, 1)

        # --- GRILLE DES VUES ---
        grid_layout = QVBoxLayout()
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        
        self.canvas_ax, self.ax_ax = self.create_view_canvas("axial")
        self.canvas_sag, self.ax_sag = self.create_view_canvas("sagittal")
        self.canvas_cor, self.ax_cor = self.create_view_canvas("coronal")
        self.canvas_empty, self.ax_empty = self.create_view_canvas("3d")
        
        row1.addWidget(self.canvas_ax); row1.addWidget(self.canvas_sag)
        row2.addWidget(self.canvas_cor); row2.addWidget(self.canvas_empty)
        
        grid_layout.addLayout(row1); grid_layout.addLayout(row2)
        main_layout.addLayout(grid_layout, 5)

        self.update_plots()
        self.showMaximized()

    def create_mitk_slider(self, name, max_val, layout, color):
        lbl = QLabel(f"{name} (Slice 1 / {max_val})")
        lbl.setStyleSheet(f"color: {color}; font-weight: bold; margin-top: 15px;")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, max_val - 1)
        slider.setValue(max_val // 2)
        slider.valueChanged.connect(self.update_plots)
        layout.addWidget(lbl)
        layout.addWidget(slider)
        return slider, lbl

    def create_view_canvas(self, view_type):
        fig = Figure(facecolor='black')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        canvas.wheelEvent = lambda event: self.handle_scroll(event, view_type)
        return canvas, ax

    def handle_scroll(self, event, view_type):
        step = 1 if event.angleDelta().y() > 0 else -1
        if view_type == "axial": self.slider_ax.setValue(self.slider_ax.value() + step)
        elif view_type == "coronal": self.slider_cor.setValue(self.slider_cor.value() + step)
        elif view_type == "sagittal": self.slider_sag.setValue(self.slider_sag.value() + step)
        event.accept()

    def reset_views(self):
        self.slider_ax.setValue(self.volume.shape[0] // 2)
        self.slider_cor.setValue(self.volume.shape[1] // 2)
        self.slider_sag.setValue(self.volume.shape[2] // 2)

    def update_plots(self):
        z, y, x = self.slider_ax.value(), self.slider_cor.value(), self.slider_sag.value()
        
        # Sécurité pour ne pas sortir des dimensions après resize potentiel
        z = min(z, self.volume.shape[0]-1)
        y = min(y, self.volume.shape[1]-1)
        x = min(x, self.volume.shape[2]-1)

        self.lbl_ax.setText(f"AXIAL (Slice {z+1} / {self.volume.shape[0]})")
        self.lbl_cor.setText(f"CORONAL (Slice {y+1} / {self.volume.shape[1]})")
        self.lbl_sag.setText(f"SAGITTAL (Slice {x+1} / {self.volume.shape[2]})")

        def draw_image(ax, canvas, data, mask_slice, title, color, lh, lv, ch, cv):
            ax.clear()
            # Affichage du volume
            ax.imshow(data, cmap='gray', aspect='equal', interpolation='nearest')
            
            # Affichage du masque par-dessus
            if mask_slice is not None and mask_slice.shape == data.shape:
                # On s'assure que le masque est binaire (0 ou 1)
                m_bin = (mask_slice > 0).astype(np.uint8)
                if np.any(m_bin):
                    ax.imshow(m_bin, cmap=self.cmap_mask, alpha=0.5, aspect='equal', interpolation='nearest', zorder=10)
            
            ax.axhline(lh, color=ch, linewidth=0.8, alpha=0.6)
            ax.axvline(lv, color=cv, linewidth=0.8, alpha=0.6)
            ax.text(0.02, 0.98, title, transform=ax.transAxes, color=color, fontweight='bold', va='top')
            ax.axis('off')
            canvas.draw()

        # Tranches Volume et Masque
        draw_image(self.ax_ax, self.canvas_ax, self.volume[z, :, :], 
                   self.mask[z, :, :] if self.mask is not None else None, 
                   "AXIAL", "#FF5555", y, x, "blue", "green")
        
        draw_image(self.ax_sag, self.canvas_sag, self.volume[:, :, x], 
                   self.mask[:, :, x] if self.mask is not None else None, 
                   "SAGITTAL", "#55FF55", z, y, "red", "blue")
        
        draw_image(self.ax_cor, self.canvas_cor, self.volume[:, y, :], 
                   self.mask[:, y, :] if self.mask is not None else None, 
                   "CORONAL", "#5555FF", z, x, "red", "green")
        
        self.ax_empty.axis('off')
        self.canvas_empty.draw()

def valider_recalage(volume_data, mask_data=None):
    # --- GESTION DES DIMENSIONS ---
    if mask_data is not None and volume_data.shape != mask_data.shape:
        print(f"Mismatch: Volume {volume_data.shape} vs Mask {mask_data.shape}. Adaptation...")
        new_mask = np.zeros(volume_data.shape, dtype=mask_data.dtype)
        z_c, y_c, x_c = [min(v, m) for v, m in zip(volume_data.shape, mask_data.shape)]
        new_mask[:z_c, :y_c, :x_c] = mask_data[:z_c, :y_c, :x_c]
        mask_data = new_mask

    # --- TRANSPOSITION ET NORMALISATION ---
    vol_mitk = np.transpose(volume_data, (0, 2, 1))
    max_v = np.max(vol_mitk) if np.max(vol_mitk) > 0 else 1.0
    data_ui = (vol_mitk / max_v * 255).astype(np.uint8)
    
    mask_ui = None
    if mask_data is not None:
        mask_ui = np.transpose(mask_data, (0, 2, 1))

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = MITKViewer(data_ui, mask_ui)
    viewer.show()

    while viewer.isVisible():
        app.processEvents()

    if QMessageBox.question(None, "Decision", "Passer à l'étape suivante ?", 
                            QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
        sys.exit()

if __name__ == "__main__":
    # Test avec un volume 320x320 et un masque 388 (pour simuler votre erreur)
    data_test = np.random.randint(0, 100, (320, 256, 256))
    mask_test = np.zeros((388, 256, 256)) # Plus grand que le volume
    mask_test[100:200, 100:200, 100:200] = 1 
    
    valider_recalage(data_test, mask_test)
>>>>>>> theirs
