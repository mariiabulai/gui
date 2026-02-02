#include "PythonBridge.h"
#include <QCoreApplication>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>  // <--- ВОТ ЭТОЙ СТРОКИ НЕ ХВАТАЛО
#include <QDebug>
#include <QFile>
#include <QMessageBox>

PythonBridge::PythonBridge(QObject *parent) : QObject(parent) {
    m_process = new QProcess(this);
    
    // Устанавливаем рабочую папку
    m_process->setWorkingDirectory("D:/GUI"); 
    
    connect(m_process, &QProcess::finished, this, &PythonBridge::onProcessFinished);
    connect(m_process, &QProcess::readyReadStandardOutput, this, &PythonBridge::onReadyReadStandardOutput);
    connect(m_process, &QProcess::readyReadStandardError, this, &PythonBridge::onReadyReadStandardError);
}

void PythonBridge::runReconstruction(const QString &imagesPath, const QString &calibrationFile) {
    // Очищаем ошибку перед запуском
    m_lastError = "";

    // 1. Формируем JSON
    QJsonObject config;
    config["images_path"] = imagesPath;
    config["calibration_file"] = calibrationFile;
    
    QString outputPly = "D:/GUI/output.ply";
    config["output_path"] = outputPly;

    // 2. Сохраняем task.json
    QString configPath = "D:/GUI/task.json";
    QFile configFile(configPath);
    if (configFile.open(QIODevice::WriteOnly)) {
        configFile.write(QJsonDocument(config).toJson());
        configFile.close();
    } else {
        emit reconstructionFinished(false, "", "Cannot create task.json");
        return;
    }

    // 3. Путь к скрипту
    QString scriptPath = "backend/main_adapter.py"; 

    // 4. Путь к Python (ИЗ VENV)
    QString pythonPath = "D:/GUI/venv/Scripts/python.exe";
    
    if (!QFile::exists(pythonPath)) {
        // Если venv не найден, пробуем системный, но предупреждаем
        qDebug() << "Venv python not found, trying 'python'...";
        pythonPath = "python"; 
    }

    QStringList args;
    args << scriptPath << configPath;
    
    emit progressUpdated("Starting Python...");
    qDebug() << "Running:" << pythonPath << args;
    
    m_process->start(pythonPath, args);
}

void PythonBridge::onReadyReadStandardOutput() {
    QByteArray data = m_process->readAllStandardOutput();
    QString text = QString::fromLocal8Bit(data).trimmed();
    
    // Дублируем в консоль разработчика
    qDebug() << "[Python]:" << text;

    // Если Python прислал текст, обновляем статус в окне!
    if (!text.isEmpty()) {
        // Берем первые 50 символов, чтобы не засорять окно
        emit progressUpdated("Py: " + text.left(50));
    }
}

void PythonBridge::onReadyReadStandardError() {
    QByteArray data = m_process->readAllStandardError();
    QString errorStr = QString::fromLocal8Bit(data);
    qDebug() << "[Python ERR]:" << errorStr;
    
    // Запоминаем ошибку
    m_lastError += errorStr;
}

void PythonBridge::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    QString outputPly = "D:/GUI/output.ply";

    if (exitCode == 0 && QFile::exists(outputPly)) {
        emit reconstructionFinished(true, outputPly, "Success");
    } else {
        // Формируем сообщение об ошибке
        QString errorMsg = "Python failed (Code " + QString::number(exitCode) + ").\n\n";
        if (!m_lastError.isEmpty()) {
            errorMsg += "Error Details:\n" + m_lastError;
        } else {
            errorMsg += "No error message received. Check Debug Console.";
        }
        
        emit reconstructionFinished(false, "", errorMsg);
    }
}