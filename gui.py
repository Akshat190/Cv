# PyQt5 libraries are used for GUI
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class ImageRestorationGuiClass(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1200, 800)  # Increased window size
        MainWindow.setStyleSheet(_fromUtf8("font: 11pt \"Dyuthi\";"))
        
        # Central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        
        # Main horizontal layout
        self.mainLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        
        # Left panel (controls) - Fixed width
        self.leftPanel = QtWidgets.QWidget()
        self.leftPanel.setFixedWidth(300)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.leftPanel)
        
        # Open/Save buttons
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.buttonOpen = QtWidgets.QPushButton("Open")
        self.buttonSave = QtWidgets.QPushButton("Save")
        self.buttonLayout.addWidget(self.buttonOpen)
        self.buttonLayout.addWidget(self.buttonSave)
        self.verticalLayout.addLayout(self.buttonLayout)
        
        # Add separator line
        self.line = QtWidgets.QFrame()
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.verticalLayout.addWidget(self.line)
        
        # Kernel selection
        self.labelBlurName = QtWidgets.QLabel("Choose Blur Kernel")
        self.comboBoxKernel = QtWidgets.QComboBox()
        self.comboBoxKernel.addItems([f"Blur Kernel {i}" for i in range(1, 5)] + ["Estimated Kernel"])
        self.labelKernelDisplay = QtWidgets.QLabel()
        self.labelKernelDisplay.setFixedSize(200, 200)
        self.verticalLayout.addWidget(self.labelBlurName)
        self.verticalLayout.addWidget(self.comboBoxKernel)
        self.verticalLayout.addWidget(self.labelKernelDisplay)
        
        # Restoration controls
        self.buttonFullInv = QtWidgets.QPushButton("Full Inverse Filter")
        self.buttonInv = QtWidgets.QPushButton("Truncated Inverse Filter")
        self.input_radius = QtWidgets.QLineEdit()
        self.input_radius.setPlaceholderText("Enter radius")
        
        self.buttonWeiner = QtWidgets.QPushButton("Wiener Filter")
        self.input_K = QtWidgets.QLineEdit()
        self.input_K.setPlaceholderText("Enter K value")
        
        self.buttonCLS = QtWidgets.QPushButton("Constrained LS Filter")
        self.input_gamma = QtWidgets.QLineEdit()
        self.input_gamma.setPlaceholderText("Enter gamma")
        
        # Add controls to layout
        for widget in [self.buttonFullInv, self.buttonInv, self.input_radius,
                      self.buttonWeiner, self.input_K, self.buttonCLS, self.input_gamma]:
            self.verticalLayout.addWidget(widget)
        
        # Add separator line for compression section
        self.compression_line = QtWidgets.QFrame()
        self.compression_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.verticalLayout.addWidget(self.compression_line)
        
        # Compression Section Label
        self.labelCompression = QtWidgets.QLabel("Image Compression")
        self.labelCompression.setStyleSheet("font-weight: bold")
        self.verticalLayout.addWidget(self.labelCompression)
        
        # Compression Method Selection
        self.compressionMethod = QtWidgets.QComboBox()
        self.compressionMethod.addItems([
            "DCT Compression",
            "Wavelet Compression",
            "SVD Compression"
        ])
        self.verticalLayout.addWidget(self.compressionMethod)
        
        # Compression Quality/Ratio Control
        self.qualityLayout = QtWidgets.QHBoxLayout()
        self.qualityLabel = QtWidgets.QLabel("Quality:")
        self.qualitySlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.qualitySlider.setMinimum(1)
        self.qualitySlider.setMaximum(100)
        self.qualitySlider.setValue(80)
        self.qualityValue = QtWidgets.QLabel("80%")
        self.qualityLayout.addWidget(self.qualityLabel)
        self.qualityLayout.addWidget(self.qualitySlider)
        self.qualityLayout.addWidget(self.qualityValue)
        self.verticalLayout.addLayout(self.qualityLayout)
        
        # Compression Filter Selection
        self.filterLabel = QtWidgets.QLabel("Pre-compression Filter:")
        self.filterCombo = QtWidgets.QComboBox()
        self.filterCombo.addItems([
            "None",
            "Gaussian Blur",
            "Median Filter",
            "Bilateral Filter"
        ])
        self.verticalLayout.addWidget(self.filterLabel)
        self.verticalLayout.addWidget(self.filterCombo)
        
        # Filter Parameters
        self.filterParamsLayout = QtWidgets.QHBoxLayout()
        self.kernelSizeLabel = QtWidgets.QLabel("Kernel:")
        self.kernelSizeInput = QtWidgets.QSpinBox()
        self.kernelSizeInput.setMinimum(3)
        self.kernelSizeInput.setMaximum(15)
        self.kernelSizeInput.setSingleStep(2)
        self.kernelSizeInput.setValue(3)
        self.filterParamsLayout.addWidget(self.kernelSizeLabel)
        self.filterParamsLayout.addWidget(self.kernelSizeInput)
        self.verticalLayout.addLayout(self.filterParamsLayout)
        
        # Compression Buttons
        self.compressButton = QtWidgets.QPushButton("Compress Image")
        self.saveCompressedButton = QtWidgets.QPushButton("Save Compressed")
        self.verticalLayout.addWidget(self.compressButton)
        self.verticalLayout.addWidget(self.saveCompressedButton)
        
        # Compression Info Labels
        self.compressionRatioLabel = QtWidgets.QLabel("Compression Ratio: --")
        self.fileSizeLabel = QtWidgets.QLabel("File Size: --")
        self.verticalLayout.addWidget(self.compressionRatioLabel)
        self.verticalLayout.addWidget(self.fileSizeLabel)
        
        # Quality assessment
        self.buttonPSNR = QtWidgets.QPushButton("Compute PSNR")
        self.buttonSSIM = QtWidgets.QPushButton("Compute SSIM")
        self.buttonTrueImage = QtWidgets.QPushButton("Load true image")
        self.buttonClearTrueImage = QtWidgets.QPushButton("Clear true image")
        
        for widget in [self.buttonPSNR, self.buttonSSIM, 
                      self.buttonTrueImage, self.buttonClearTrueImage]:
            self.verticalLayout.addWidget(widget)
        
        # Right panel (images)
        self.rightPanel = QtWidgets.QWidget()
        self.rightLayout = QtWidgets.QVBoxLayout(self.rightPanel)
        
        # Image labels
        self.imageLayout = QtWidgets.QHBoxLayout()
        self.labelIn = QtWidgets.QLabel()
        self.labelOut = QtWidgets.QLabel()
        
        # Set fixed size for image labels
        for label in [self.labelIn, self.labelOut]:
            label.setMinimumSize(400, 400)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid gray")
        
        self.imageLayout.addWidget(self.labelIn)
        self.imageLayout.addWidget(self.labelOut)
        
        # Image titles
        self.titleLayout = QtWidgets.QHBoxLayout()
        self.labelInTitle = QtWidgets.QLabel("Original Image")
        self.labelOutTitle = QtWidgets.QLabel("Restored Image")
        self.titleLayout.addWidget(self.labelInTitle, alignment=QtCore.Qt.AlignCenter)
        self.titleLayout.addWidget(self.labelOutTitle, alignment=QtCore.Qt.AlignCenter)
        
        # Metrics display
        self.metricsLayout = QtWidgets.QHBoxLayout()
        self.leftMetrics = QtWidgets.QVBoxLayout()
        self.rightMetrics = QtWidgets.QVBoxLayout()
        
        # PSNR/SSIM labels
        self.label_og_psnr = QtWidgets.QLabel("PSNR: --")
        self.label_og_ssim = QtWidgets.QLabel("SSIM: --")
        self.label_res_psnr = QtWidgets.QLabel("PSNR: --")
        self.label_res_ssim = QtWidgets.QLabel("SSIM: --")
        
        self.leftMetrics.addWidget(self.label_og_psnr)
        self.leftMetrics.addWidget(self.label_og_ssim)
        self.rightMetrics.addWidget(self.label_res_psnr)
        self.rightMetrics.addWidget(self.label_res_ssim)
        
        self.metricsLayout.addLayout(self.leftMetrics)
        self.metricsLayout.addLayout(self.rightMetrics)
        
        # Add layouts to right panel
        self.rightLayout.addLayout(self.titleLayout)
        self.rightLayout.addLayout(self.imageLayout)
        self.rightLayout.addLayout(self.metricsLayout)
        
        # Add panels to main layout
        self.mainLayout.addWidget(self.leftPanel)
        self.mainLayout.addWidget(self.rightPanel)
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        # Menu and status bars
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Restoration", None))

