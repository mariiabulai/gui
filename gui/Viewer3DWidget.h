#pragma once
#include <QWidget>
#include <QDebug>

class Viewer3DWidget : public QWidget
{
    Q_OBJECT
public:
    explicit Viewer3DWidget(QWidget *parent = nullptr) : QWidget(parent) {}

public slots:
    // Головний метод, який викликає Controller
    void loadPlyModel(const QString& plyFilePath) 
    {
        qDebug() << "Viewer3DWidget: Received command to load PLY model from path:" << plyFilePath;
        // Тут має бути справжня логіка QOpenGLWidget / Qt3D
    }
};