#include "controller.h"
#include <QMessageBox>
#include <QDir>
#include <QStandardPaths>
#include <QDebug>

Controller::Controller(Window* window, QObject* parent) 
    : QObject(parent), m_window(window) 
{
    m_bridge = new PythonBridge(this);

    connect(m_window, &Window::startRequested, this, &Controller::startReconstruction);
    connect(m_bridge, &PythonBridge::reconstructionFinished, this, &Controller::handleReconstructionResult);
    connect(m_bridge, &PythonBridge::progressUpdated, this, [this](const QString &msg){
        m_window->updateStatus(msg);
    });
}

void Controller::startReconstruction() {
    // 1. Спрашиваем калибровку
    QString calibPath = m_window->promptForCalibrationFile();
    if (calibPath.isEmpty()) return; 

    // 2. Создаем ЧИСТУЮ папку для фото
    // Важно: используем абсолютный путь, чтобы не потеряться
    QString tempPath = QDir::current().absoluteFilePath("staging_images");
    QDir dir(tempPath);
    
    // УДАЛЯЕМ старую папку со всем содержимым (чтобы не было фото шахматной доски)
    if (dir.exists()) {
        dir.removeRecursively();
    }
    // Создаем новую пустую
    if (!dir.mkpath(".")) {
        QMessageBox::critical(m_window, "Error", "Cannot create temp folder:\n" + tempPath);
        return;
    }

    // 3. Копируем выбранные пользователем фото
    QStringList images = m_window->getSelectedImages();
    if (images.isEmpty()) {
        QMessageBox::warning(m_window, "Error", "No images selected in the list!");
        return;
    }

    m_window->updateStatus("Copying images...", 0);
    
    int copied = 0;
    for (const QString &srcPath : images) {
        QString fileName = QFileInfo(srcPath).fileName();
        QString destPath = dir.filePath(fileName);
        
        if (QFile::copy(srcPath, destPath)) {
            copied++;
        } else {
            qDebug() << "Failed to copy:" << srcPath;
        }
    }
    
    qDebug() << "Copied" << copied << "images to" << tempPath;

    // 4. Запуск
    m_window->updateStatus("Starting Python backend...", 10);
    m_window->setControlsEnabled(false);
    
    // Передаем путь именно к ЭТОЙ временной папке
    m_bridge->runReconstruction(tempPath, calibPath);
}

void Controller::handleReconstructionResult(bool success, const QString& plyPath, const QString& msg) {
    m_window->setControlsEnabled(true);
    
    if (success) {
        m_window->updateStatus("Done!", 100);
        m_window->show3DModel(plyPath);
    } else {
        m_window->updateStatus("Error", 0);
        QMessageBox::critical(m_window, "Reconstruction Failed", msg);
    }
}