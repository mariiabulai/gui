#pragma once
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <vector>

struct Point3D {
    float x, y, z;
    float r, g, b;
};

class Viewer3DWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    explicit Viewer3DWidget(QWidget *parent = nullptr);
    void loadPly(const QString &filename);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    
    void mousePressEvent(QMouseEvent *e) override;
    void mouseMoveEvent(QMouseEvent *e) override;
    void wheelEvent(QWheelEvent *e) override;

private:
    std::vector<Point3D> m_points;
    float m_rotX = 0, m_rotY = 0;
    float m_zoom = -10.0f;
    QPoint m_lastMousePos;
};