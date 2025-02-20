# ---------------------------------------------------------------#
# __name__ = "ImageRestoration_EE610_Assignment"
# __author__ = "Shyama P"
# __version__ = "1.0"
# __email__ = "183079031@iitb.ac.in"
# __status__ = "Development"
# ---------------------------------------------------------------#

# main.py contains the code for initializing and running the code for GUI
import sys
# PyQt5 libraries are used for GUI
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# OpenCV2 library is used for reading/ writing of images
import cv2
# All array operations are performed using numpy library
import numpy as np
from scipy.fftpack import dct, idct
import pywt

# The GUI structure definition is provided in gui.py
from gui import *
# Image restoration logic is defined in imageRestorationFns.py
import imageRestorationFns as ir
# Image compression logic is defined in imageCompressionFns.py
import imageCompressionFns as ic


# class ImageEditorClass implements the GUI main window class
class ImageRestorationClass(QMainWindow):

    # stores a copy of original image for use in Undo All functionality
    originalImage = [0]
    # stores the current image being displayed/ processed
    currentImage = [0]
    # stores the ground truth image for psnr/ ssim calculations
    trueImage = [0]

    # stores current image height and width
    imageWidth = 0
    imageHeight = 0

    # GUI initialization
    def __init__(self, parent=None):
        super(ImageRestorationClass, self).__init__()
        self.ui = ImageRestorationGuiClass()
        self.ui.setupUi(self)
        
        # Set window size and position
        screen = QDesktopWidget().screenGeometry()
        self.setGeometry(100, 100, int(screen.width() * 0.8), int(screen.height() * 0.8))
        
        # Connect buttons
        self.ui.buttonOpen.clicked.connect(self.open_image)
        self.ui.buttonSave.clicked.connect(self.save_image)
        self.ui.buttonFullInv.clicked.connect(self.call_full_inverse)
        self.ui.buttonInv.clicked.connect(self.call_truncated_inverse_filter)
        self.ui.buttonWeiner.clicked.connect(self.call_weiner_filter)
        self.ui.buttonCLS.clicked.connect(self.call_constrained_ls_filter)
        self.ui.buttonPSNR.clicked.connect(self.calculate_psnr)
        self.ui.buttonSSIM.clicked.connect(self.calculate_ssim)
        self.ui.buttonTrueImage.clicked.connect(self.set_true_image)
        self.ui.buttonClearTrueImage.clicked.connect(self.reset_true_image)
        
        self.ui.comboBoxKernel.currentIndexChanged.connect(self.displayKernel)
        
        # Connect compression controls
        self.ui.qualitySlider.valueChanged.connect(self.updateQualityLabel)
        self.ui.compressButton.clicked.connect(self.compressImage)
        self.ui.saveCompressedButton.clicked.connect(self.saveCompressedImage)
        self.ui.filterCombo.currentIndexChanged.connect(self.updateFilterParams)
        
        self.disableAll()

    # calls the full inverse function
    def call_full_inverse(self):
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            self.currentImage = ir.full_inverse_filter(self.originalImage, blur_kernel)
            self.displayOutputImage()

            # compute psnr and ssim for output if true image is available
            if not np.array_equal(self.trueImage, np.array([0])):
                self.calculate_psnr()
                self.calculate_ssim()

    # calls the truncated inverse function
    def call_truncated_inverse_filter(self):
        self.ui.input_radius.setStyleSheet("background-color: white;")
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            # read the blur kernel radius from line edit input object
            R = self.ui.input_radius.text()
            if R and float(R) > 0:
                radius = float(R)
                self.currentImage = ir.truncated_inverse_filter(self.originalImage, blur_kernel, radius)
                self.displayOutputImage()

                # compute psnr and ssim for output if true image is available
                if not np.array_equal(self.trueImage, np.array([0])):
                    self.calculate_psnr()
                    self.calculate_ssim()

            else:
                self.ui.input_radius.setStyleSheet("background-color: red;")

    # calls the weiner filter function
    def call_weiner_filter(self):
        self.ui.input_K.setStyleSheet("background-color: white;")
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            # read the K value from line edit input object
            K_str = self.ui.input_K.text()
            if K_str:
                K = float(K_str)
                self.currentImage = ir.weiner_filter(self.originalImage, blur_kernel, K)
                self.displayOutputImage()

                # compute psnr and ssim for output if true image is available
                if not np.array_equal(self.trueImage, np.array([0])):
                    self.calculate_psnr()
                    self.calculate_ssim()

            else:
                self.ui.input_K.setStyleSheet("background-color: red;")

    # calls the constrained ls function
    def call_constrained_ls_filter(self):
        self.ui.input_gamma.setStyleSheet("background-color: white;")
        if not np.array_equal(self.originalImage, np.array([0])):
            # read the selected blur kernel
            blur_kernel = self.get_blur_kernel()
            # read the gamma value from line edit input object
            Y = self.ui.input_gamma.text()
            if Y:
                gamma = float(Y)
                self.currentImage = ir.constrained_ls_filter(self.originalImage, blur_kernel, gamma)
                self.displayOutputImage()

                # compute psnr and ssim for output if true image is available
                if not np.array_equal(self.trueImage, np.array([0])):
                    self.calculate_psnr()
                    self.calculate_ssim()
            else:
                self.ui.input_gamma.setStyleSheet("background-color: red;")

    # calls the compute ssim function for input and output image
    def calculate_ssim(self):
        if not (np.array_equal(self.originalImage, np.array([0])) or np.array_equal(self.currentImage, np.array([0]))):

            if np.array_equal(self.trueImage, np.array([0])):
                self.set_true_image()

            if not np.array_equal(self.trueImage, np.array([0])):
                # compute ssim for input and output images
                ssim_in = ir.ssim(self.trueImage, self.originalImage)
                ssim_out = ir.ssim(self.trueImage, self.currentImage)
                # display ssim values
                self.ui.label_og_ssim.setText(str(ssim_in))
                self.ui.label_res_ssim.setText(str(ssim_out))

    # calls the compute psnr function for input and output image
    def calculate_psnr(self):
        if not (np.array_equal(self.originalImage, np.array([0])) or np.array_equal(self.currentImage, np.array([0]))):

            if np.array_equal(self.trueImage, np.array([0])):
                self.set_true_image()

            if not np.array_equal(self.trueImage, np.array([0])):
                # compute psnr for input and output images
                psnr_in = ir.psnr(self.trueImage, self.originalImage)
                psnr_out = ir.psnr(self.trueImage, self.currentImage)
                # display psnr values
                self.ui.label_og_psnr.setText(str(psnr_in))
                self.ui.label_res_psnr.setText(str(psnr_out))

    # open true image file
    def set_true_image(self):
        if not (np.array_equal(self.originalImage, np.array([0])) or np.array_equal(self.currentImage, np.array([0]))):
            # open a new Open Image dialog box to select original image
            open_image_window = QFileDialog()
            image_path, _ = QFileDialog.getOpenFileName(open_image_window, 'Select original image', '/')

            # check if image path is not null or empty
            if image_path:
                # read original image
                self.trueImage = cv2.imread(image_path, 1)

    # clear the current true image
    def reset_true_image(self):
        self.trueImage = [0]
        self.ui.label_og_psnr.setText('--')
        self.ui.label_res_psnr.setText('--')
        self.ui.label_og_ssim.setText('--')
        self.ui.label_res_ssim.setText('--')

    # read the selected blur kernel from kernels folder
    def get_blur_kernel(self):
        index = self.ui.comboBoxKernel.currentIndex()
        kernel_filename = 'kernels/' + str(index + 1) + '.bmp'
        kernel = np.array(cv2.imread(kernel_filename, 0))
        return kernel

    # called when Open button is clicked
    def open_image(self):
        # open a new Open Image dialog box and capture path of file selected
        open_image_window = QFileDialog()
        image_path, _ = QFileDialog.getOpenFileName(open_image_window, 'Open Image', '/')  # Unpack tuple

        # check if image path is not null or empty
        if image_path:
            # initialize class variables
            self.currentImage = [0]
            self.trueImage = [0]

            # read image at selected path to a numpy ndarray object as color image
            self.currentImage = cv2.imread(image_path, 1)

            # set image specific class variables based on current image
            self.imageWidth = self.currentImage.shape[1]
            self.imageHeight = self.currentImage.shape[0]

            self.originalImage = self.currentImage.copy()

            # displayInputImage converts original image from ndarry format to
            # pixmap and assigns it to image display label
            self.displayInputImage()

            self.ui.labelOut.clear()

            # Enable all buttons and sliders
            self.enableAll()

    # called when Save button is clicked
    def save_image(self):
        # configure the save image dialog box to use .jpg extension for image if
        # not provided in file name
        dialog = QFileDialog()
        dialog.setDefaultSuffix('jpg')
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        # open the save dialog box and wait until user clicks 'Save'
        # button in the dialog box
        if dialog.exec_() == QDialog.Accepted:
            # select the first path in the selected files list as image save
            # location
            save_image_filename = dialog.selectedFiles()[0]
            # write current image to the file path selected by user
            cv2.imwrite(save_image_filename, self.currentImage)

    # displayInputImage converts original image from ndarry format to pixmap and
    # assigns it to input image display label
    def displayInputImage(self):
        # set display size to size of the image display label
        display_size = self.ui.labelIn.size()
        # copy original image to temporary variable for processing pixmap
        image = np.array(self.originalImage.copy())
        zero = np.array([0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            # convert BGR image to RGB format for display in label
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # ndarray cannot be directly converted to QPixmap format required
            # by image display label
            # so ndarray is first converted to QImage and then QImage to QPixmap
            # convert image ndarray to QImage format
            qImage = QImage(image, self.imageWidth, self.imageHeight,
                            self.imageWidth * 3, QImage.Format_RGB888)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap.fromImage(qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            # set pixmap to image display label in GUI
            self.ui.labelIn.setPixmap(pixmap)

    # displayOutputImage converts current image from ndarry format to pixmap and
    # assigns it to output image display label
    def displayOutputImage(self):
        # set display size to size of the image display label
        display_size = self.ui.labelOut.size()
        # copy current image to temporary variable for processing pixmap
        image = np.array(self.currentImage.copy())
        zero = np.array([0])

        # display image if image is not [0] array
        if not np.array_equal(image, zero):
            # convert BGR image to RGB format for display in label
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # ndarray cannot be directly converted to QPixmap format required
            # by image display label
            # so ndarray is first converted to QImage and then QImage to QPixmap
            # convert image ndarray to QImage format
            qImage = QImage(image, self.imageWidth, self.imageHeight,
                            self.imageWidth * 3, QImage.Format_RGB888)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap.fromImage(qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            # set pixmap to image display label in GUI
            self.ui.labelOut.setPixmap(pixmap)

    # displayKernel converts selected kernel image from ndarry format to pixmap and
    # assigns it to kernel display label
    def displayKernel(self):
        # set display size to size of the kernel display label
        display_size = self.ui.labelKernelDisplay.size()
        # copy kernel image to temporary variable for processing pixmap
        kernel = np.array(self.get_blur_kernel())
        zero = np.array([0])

        # display image if kernel is not [0] array
        if not np.array_equal(kernel, zero):
            # ndarray cannot be directly converted to QPixmap format required
            # by kernel display label
            # so ndarray is first converted to QImage and then QImage to QPixmap

            # convert kernel ndarray to QImage format
            qImage = QImage(kernel, kernel.shape[1], kernel.shape[0], kernel.shape[1], QImage.Format_Indexed8)

            # convert QImage to QPixmap for loading in image display label
            pixmap = QPixmap.fromImage(qImage)
            pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
            # set pixmap to kernel display label in GUI
            self.ui.labelKernelDisplay.setPixmap(pixmap)

    # Function to enable all buttons and sliders
    def enableAll(self):
        self.ui.buttonSave.setEnabled(True)
        self.ui.buttonFullInv.setEnabled(True)
        self.ui.buttonInv.setEnabled(True)
        self.ui.buttonWeiner.setEnabled(True)
        self.ui.buttonCLS.setEnabled(True)
        self.ui.buttonPSNR.setEnabled(True)
        self.ui.buttonSSIM.setEnabled(True)
        self.ui.buttonTrueImage.setEnabled(True)
        self.ui.buttonClearTrueImage.setEnabled(True)
        self.ui.comboBoxKernel.setEnabled(True)
        self.ui.input_radius.setEnabled(True)
        self.ui.input_K.setEnabled(True)
        self.ui.input_gamma.setEnabled(True)

        self.ui.input_radius.clear()
        self.ui.input_K.clear()
        self.ui.input_gamma.clear()

        self.ui.label_og_psnr.setText('--')
        self.ui.label_res_psnr.setText('--')
        self.ui.label_og_ssim.setText('--')
        self.ui.label_res_ssim.setText('--')

    # Function to disable all buttons and sliders
    def disableAll(self):
        self.ui.buttonSave.setEnabled(False)
        self.ui.buttonFullInv.setEnabled(False)
        self.ui.buttonInv.setEnabled(False)
        self.ui.buttonWeiner.setEnabled(False)
        self.ui.buttonCLS.setEnabled(False)
        self.ui.buttonPSNR.setEnabled(False)
        self.ui.buttonSSIM.setEnabled(False)
        self.ui.buttonTrueImage.setEnabled(False)
        self.ui.buttonClearTrueImage.setEnabled(False)
        self.ui.comboBoxKernel.setEnabled(False)
        self.ui.input_radius.setEnabled(False)
        self.ui.input_K.setEnabled(False)
        self.ui.input_gamma.setEnabled(False)

        self.ui.input_radius.clear()
        self.ui.input_K.clear()
        self.ui.input_gamma.clear()

        self.ui.label_og_psnr.setText('--')
        self.ui.label_res_psnr.setText('--')
        self.ui.label_og_ssim.setText('--')
        self.ui.label_res_ssim.setText('--')

    def updateQualityLabel(self):
        value = self.ui.qualitySlider.value()
        self.ui.qualityValue.setText(f"{value}%")
        
    def updateFilterParams(self):
        filter_type = self.ui.filterCombo.currentText()
        self.ui.kernelSizeInput.setEnabled(filter_type != "None")
        
    def applyPreFilter(self, image):
        filter_type = self.ui.filterCombo.currentText()
        kernel_size = self.ui.kernelSizeInput.value()
        
        if filter_type == "None":
            return image
        elif filter_type == "Gaussian Blur":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif filter_type == "Median Filter":
            return cv2.medianBlur(image, kernel_size)
        elif filter_type == "Bilateral Filter":
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        
    def compressImage(self):
        if self.originalImage is None:
            return
            
        # Apply pre-compression filter
        filtered_image = self.applyPreFilter(self.originalImage)
        
        # Get compression parameters
        method = self.ui.compressionMethod.currentText()
        quality = self.ui.qualitySlider.value()
        
        if method == "DCT Compression":
            self.currentImage = self.dctCompress(filtered_image, quality)
        elif method == "Wavelet Compression":
            self.currentImage = self.waveletCompress(filtered_image, quality)
        elif method == "SVD Compression":
            self.currentImage = self.svdCompress(filtered_image, quality)
            
        # Update compression info
        original_size = self.originalImage.nbytes
        compressed_size = self.currentImage.nbytes
        ratio = original_size / compressed_size
        
        self.ui.compressionRatioLabel.setText(f"Compression Ratio: {ratio:.2f}:1")
        self.ui.fileSizeLabel.setText(f"File Size: {compressed_size/1024:.1f} KB")
        
        # Display compressed image
        self.displayOutputImage()
        
    def dctCompress(self, image, quality):
        # DCT compression implementation
        blocks = []
        for i in range(0, image.shape[0], 8):
            for j in range(0, image.shape[1], 8):
                block = image[i:i+8, j:j+8]
                if block.shape[0] != 8 or block.shape[1] != 8:
                    continue
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                threshold = np.percentile(np.abs(dct_block), 100-quality)
                dct_block[np.abs(dct_block) < threshold] = 0
                blocks.append((i, j, dct_block))
                
        result = np.zeros_like(image)
        for i, j, block in blocks:
            result[i:i+8, j:j+8] = idct(idct(block.T, norm='ortho').T, norm='ortho')
            
        return result
        
    def waveletCompress(self, image, quality):
        # Wavelet compression implementation
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs
        
        # Threshold based on quality
        threshold = np.percentile(np.abs(cH), 100-quality)
        cH[np.abs(cH) < threshold] = 0
        cV[np.abs(cV) < threshold] = 0
        cD[np.abs(cD) < threshold] = 0
        
        return pywt.idwt2((cA, (cH, cV, cD)), 'haar')
        
    def svdCompress(self, image, quality):
        # SVD compression implementation
        U, s, Vt = np.linalg.svd(image)
        k = int((quality/100.0) * len(s))
        compressed = np.dot(U[:, :k], np.dot(np.diag(s[:k]), Vt[:k, :]))
        return np.clip(compressed, 0, 255).astype(np.uint8)
        
    def saveCompressedImage(self):
        if self.currentImage is None:
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Compressed Image',
            '', 'JPEG (*.jpg);;PNG (*.png)'
        )
        if filename:
            cv2.imwrite(filename, self.currentImage)


# initialize the ImageEditorClass and run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = ImageRestorationClass()
    myapp.showMaximized()
    sys.exit(app.exec_())

