#include "PythonBridge.h"
#include <QDebug>

void PythonBridge::runReconstruction(const QStringList& imagePaths)
{
    qDebug() << "PythonBridge: Starting MOCK reconstruction for" << imagePaths.size() << "images.";
    
    // Імітуємо запуск QProcess
    QTimer::singleShot(2000, this, &PythonBridge::mockReconstructionComplete); // Затримка 2 секунди
}

void PythonBridge::mockReconstructionComplete()
{
    qDebug() << "PythonBridge: MOCK reconstruction finished successfully.";
    
    // Імітуємо повернення шляху до PLY-файлу
    QFileInfo plyFile(m_mockPlyPath);
    
    // Посилаємо сигнал успіху назад у AppController
    emit backendTaskFinished(true, plyFile, ""); 
}