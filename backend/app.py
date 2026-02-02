import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image 

# --- Funkcje przetwarzania obrazu (Computer Vision) ---

def find_matches(img1, img2):
    """Znajduje dopasowania punktów kluczowych między dwoma obrazami."""
    
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    if des1 is None or des2 is None:
        return None, None, None
        
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = matches 
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    return pts1, pts2, img_matches

def reconstruct_3d(pts1, pts2, K):
    """
    Odtwarza strukturę 3D (RZADKĄ) i zwraca parametry kamery.
    """
    
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)
    
    if mask is None:
        return None, None, None, None

    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    
    E = K.T @ F @ K
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
    points_3d = (points_4d_hom / points_4d_hom[3]).T[:, :3]
    
    return points_3d, R, t, mask_pose

def create_dense_cloud(img1_gray, img2_gray, K, R, t, img_size):
    """
    Tworzy GĘSTĄ chmurę punktów używając Stereo Matching.
    """
    w, h = img_size
    
    distCoeffs = np.zeros(4)
    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, distCoeffs, K, distCoeffs, (w,h), R, t, alpha=1
    )
    
    map1x, map1y = cv2.initUndistortRectifyMap(K, distCoeffs, R1, P1, (w,h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, distCoeffs, R2, P2, (w,h), cv2.CV_32FC1)
    
    img1_rect = cv2.remap(img1_gray, map1x, map1y, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2_gray, map2x, map2y, cv2.INTER_LINEAR)
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 6,  # 96 dysparacji
        blockSize=7,            # Zwiększamy blockSize
        P1=8 * 3 * 7**2,
        P2=32 * 3 * 7**2,
        disp12MaxDiff=10,        # Zwiększamy dozwoloną różnicę
        uniquenessRatio=5,      # Obniżamy wymaganą unikalność
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity_map = stereo.compute(img1_rect, img2_rect).astype(np.float32) / 16.0
    
    points_3d_dense = cv2.reprojectImageTo3D(disparity_map, Q)
    
    return points_3d_dense, disparity_map

# --- Funkcje dla GUI (Streamlit) ---

def get_3d_plot(points_3d, is_dense=False):
    """
    Tworzy interaktywny wykres 3D Plotly. 
    --- NOWA, ULEPSZONA WERSJA FILTROWANIA ---
    """
    
    if is_dense:
        plot_title_base = "Odtworzona GĘSTA chmura punktów"
    else:
        plot_title_base = "Odtworzona RZADKA chmura punktów"

    # 1. Usunięcie nieskończoności (inf)
    mask_finite = np.isfinite(points_3d).all(axis=1)
    points_3d = points_3d[mask_finite]

    if points_3d.shape[0] == 0:
        return go.Figure(layout=go.Layout(title=f"{plot_title_base} (Brak danych po filtracji)"))

    # 2. Filtrowanie ekstremalnych wartości (outlierów) na osi Z
    # Używamy percentyli, aby być odpornym na dziwną skalę
    z_vals = points_3d[:, 2]
    # Bierzemy zakres od 5 do 95 percentyla
    z_min = np.percentile(z_vals, 5)
    z_max = np.percentile(z_vals, 95)
    
    # Dodajemy mały margines
    margin = (z_max - z_min) * 0.1 
    
    mask_z = (z_vals > (z_min - margin)) & (z_vals < (z_max + margin))
    points_3d = points_3d[mask_z]

    if points_3d.shape[0] == 0:
         return go.Figure(layout=go.Layout(title=f"{plot_title_base} (Brak danych po filtracji)"))

    # 3. Próbkowanie (Downsampling)
    if is_dense and points_3d.shape[0] > 100000:
        step = 50 
        points_3d = points_3d[::step]
        plot_title = f"{plot_title_base} (co {step}-ty punkt)"
    else:
        plot_title = plot_title_base


    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=plot_title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(aspectmode='data')
    )
    return fig

# --- Główna aplikacja Streamlit ---

st.set_page_config(layout="wide", page_title="Rekonstrukcja 3D")
st.title("📸 → 🧊 Projekt: Rekonstrukcja sceny 3D z obrazów 2D")

st.sidebar.header("Panel sterowania")
img1_file = st.sidebar.file_uploader("Wczytaj obraz 1", type=["jpg", "png", "jpeg"])
img2_file = st.sidebar.file_uploader("Wczytaj obraz 2", type=["jpg", "png", "jpeg"])

st.sidebar.markdown("---")
run_dense = st.sidebar.checkbox("Generuj GĘSTĄ chmurę (wolno!)", value=False)
st.sidebar.markdown(
    "> *Gęsta chmura używa Stereo Matchingu do znalezienia głębi "
    "dla każdego piksela. Działa najlepiej dla zdjęć "
    "przesuniętych tylko w bok (bez obrotu).*")
st.sidebar.markdown("---")

run_button = st.sidebar.button("🚀 Uruchom rekonstrukcję")

if run_button and img1_file and img2_file:
    
    img1_pil = Image.open(img1_file)
    img2_pil = Image.open(img2_file)
    
    img1_color = np.array(img1_pil)
    img2_color = np.array(img2_pil)
    
    gray1 = cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_color, cv2.COLOR_RGB2GRAY)
    
    h, w, _ = img1_color.shape
    st.session_state.img_size = (w, h)
    st.session_state.K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)

    with st.spinner('Krok 1: Wyszukiwanie dopasowań (ORB)...'):
        pts1, pts2, img_matches = find_matches(gray1, gray2)
        if pts1 is None:
            st.error("Nie znaleziono wystarczającej liczby dopasowań. Spróbuj innych zdjęć.")
        else:
            st.session_state.img_matches = img_matches
            st.session_state.pts1 = pts1
            st.session_state.pts2 = pts2

    with st.spinner('Krok 2: Triangulacja RZADKA (RANSAC + recoverPose)...'):
        points_3d_sparse, R, t, mask = reconstruct_3d(
            st.session_state.pts1, st.session_state.pts2, st.session_state.K
        )
        if points_3d_sparse is None:
             st.error("Nie udało się odtworzyć pozy kamery (RANSAC zawiódł). Spróbuj innych zdjęć.")
        else:
            st.session_state.points_3d_sparse = points_3d_sparse
            st.session_state.R = R
            st.session_state.t = t
            st.success("✅ Rekonstrukcja RZADKA zakończona!")

    if run_dense and 'R' in st.session_state:
        st.warning("Uruchomiono rekonstrukcję GĘSTĄ. To może zająć 1-2 minuty...")
        with st.spinner('Krok 3: Obliczanie mapy dysparacji (StereoSGBM)...'):
            points_3d_dense, disparity_map = create_dense_cloud(
                gray1, gray2, 
                st.session_state.K, 
                st.session_state.R, 
                st.session_state.t, 
                st.session_state.img_size
            )
            
            st.session_state.points_3d_dense_flat = points_3d_dense.reshape(-1, 3)
            st.session_state.disparity_map = disparity_map
            st.success("✅ Rekonstrukcja GĘSTA zakończona!")


# --- Zakładki z wynikami ---
tab1_title = "[Krok 1] Dopasowywanie punktów"
tab2_title = "[Krok 2] Chmura RZADKA"
tab3_title = "[Krok 3] Chmura GĘSTA"

tab1, tab2, tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

with tab1:
    st.header("Dopasowywanie punktów kluczowych")
    if 'img_matches' in st.session_state:
        st.image(st.session_state.img_matches, use_container_width=True, caption="50 najlepszych dopasowań (ORB)")
    else:
        st.info("Wczytaj dwa obrazy i kliknij 'Uruchom rekonstrukcję'.")

with tab2:
    st.header("Interaktywna chmura punktów 3D (RZADKA)")
    if 'points_3d_sparse' in st.session_state:
        fig_sparse = get_3d_plot(st.session_state.points_3d_sparse, is_dense=False)
        st.plotly_chart(fig_sparse, use_container_width=True)
    else:
        st.info("Wyniki rekonstrukcji RZADKIEJ pojawią się tutaj.")

with tab3:
    st.header("Interaktywna chmura punktów 3D (GĘSTA)")
    if 'points_3d_dense_flat' in st.session_state:
        
        st.subheader("Mapa Dysparacji (Mapa Głębokości)")
        
        # --- POPRAWIONA WIZUALIZACJA MAPY ---
        raw_map = st.session_state.disparity_map.copy()
        
        # Ustawiamy nieprawidłowe wartości (ujemne) na 0, aby nie psuły skali
        raw_map[raw_map < 0] = 0
        
        # Teraz normalizujemy tylko prawidłowy zakres [0, max_disparity] do [0, 255]
        normalized_map = cv2.normalize(
            src=raw_map, 
            dst=None, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        st.image(normalized_map, caption="Mapa dysparacji (jaśniejsze = bliżej)")
        # --- KONIEC POPRAWKI ---
        
        st.markdown("---")
        
        st.subheader("Wykres 3D chmury gęstej")
        fig_dense = get_3d_plot(st.session_state.points_3d_dense_flat, is_dense=True)
        st.plotly_chart(fig_dense, use_container_width=True)
    else:
        st.info("Wyniki rekonstrukcji GĘSTEJ pojawią się tutaj po zaznaczeniu opcji i uruchomieniu.")