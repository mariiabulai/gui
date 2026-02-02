#include "window.h"
#include "ui_window.h" 
#include <QFileDialog>
#include <QLayout>
#include <QMessageBox>

Window::Window(QWidget *parent) : QMainWindow(parent), ui(new Ui::window) {
    ui->setupUi(this);

    // Встраиваем наш кастомный OpenGL виджет в контейнер из UI
    QVBoxLayout* layout = new QVBoxLayout(ui->viewer3DWidget);
    layout->setContentsMargins(0,0,0,0);
    m_viewer = new Viewer3DWidget(ui->viewer3DWidget);
    layout->addWidget(m_viewer);
}

Window::~Window() {
    delete ui;
}

QStringList Window::getSelectedImages() const {
    QStringList paths;
    for(int i = 0; i < ui->imageListWidget->count(); ++i) {
        paths.append(ui->imageListWidget->item(i)->data(Qt::UserRole).toString());
    }
    return paths;
}

void Window::updateStatus(const QString &msg, int progress) {
    this->setWindowTitle("Reconstruction: " + msg);
    if (progress >= 0) {
        ui->progressBar->setValue(progress);
    }
}

void Window::show3DModel(const QString &plyPath) {
    m_viewer->loadPly(plyPath);
}

void Window::setControlsEnabled(bool enable) {
    ui->startButton->setEnabled(enable);
    ui->addPhotosButton->setEnabled(enable);
    ui->removePhotoButton->setEnabled(enable);
}

// Ось ця функція повинна бути тільки ОДИН РАЗ
QString Window::promptForCalibrationFile() {
    return QFileDialog::getOpenFileName(this, 
        "Select Calibration File", 
        "", 
        "NPZ Files (*.npz);;All Files (*)");
}

// --- Слоты кнопок ---

void Window::on_addPhotosButton_clicked() {
    QStringList files = QFileDialog::getOpenFileNames(this, 
        "Select Photos", 
        "", 
        "Images (*.png *.jpg *.jpeg)");
        
    for(const QString &file : files) {
        bool exists = false;
        for(int i=0; i<ui->imageListWidget->count(); ++i) {
            if (ui->imageListWidget->item(i)->data(Qt::UserRole).toString() == file) {
                exists = true; 
                break;
            }
        }
        
        if (!exists) {
            QListWidgetItem *item = new QListWidgetItem(QFileInfo(file).fileName());
            item->setData(Qt::UserRole, file); 
            ui->imageListWidget->addItem(item);
        }
    }
}

void Window::on_removePhotoButton_clicked() {
    qDeleteAll(ui->imageListWidget->selectedItems());
}

void Window::on_startButton_clicked() {
    if (ui->imageListWidget->count() < 2) {
        QMessageBox::warning(this, "Warning", "Please add at least 2 photos.");
        return;
    }
    emit startRequested();
}