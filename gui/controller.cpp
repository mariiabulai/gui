#include "controller.h"
#include <QTimer>
#include <QDebug>

controller::controller(QObject *parent) : QObject(parent) {
}

void controller::startReconstruction(const QStringList& images) {
    // 1. Кажемо GUI заблокуватися
    emit reconstructionStarted();

    qDebug() << "Start Simulation...";
    qDebug() << "Images:" << images.size();

    // 2. Імітуємо бурхливу діяльність (3 секунди)
    QTimer::singleShot(3000, this, [this]() {
        qDebug() << "Simulation Done!";
        // 3. Кажемо GUI, що все готово
        emit reconstructionFinished();
    });
}