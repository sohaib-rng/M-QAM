import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas as pd

# Fonction pour générer la constellation QAM
def generate_qam_constellation(M, exclude_points):
    side_len = int(np.sqrt(M))
    if M == 8:
        side_len_x = 4
        side_len_y = 2
    elif M == 32:
        side_len_x = 6
        side_len_y = 6
    elif M == 128:
        side_len_x = 12
        side_len_y = 12
    elif M == 512:
        side_len_x = 24
        side_len_y = 24
    elif M == 2048:
        side_len_x = 46
        side_len_y = 46
    else:
        side_len_x = side_len
        side_len_y = side_len

    points = np.array([(x, y) for x in range(-side_len_x + 1, side_len_x, 2)
                       for y in range(-side_len_y + 1, side_len_y, 2)])
    points = points[~np.any(points == 0, axis=1)]

    excluded_points = []
    if exclude_points > 0:
        square_size = int(np.sqrt(exclude_points // 4))
        for i in range(square_size):
            for j in range(square_size):
                excluded_points.extend([
                    (side_len_x - 1 - 2 * i, side_len_y - 1 - 2 * j),
                    (-side_len_x + 1 + 2 * i, side_len_y - 1 - 2 * j),
                    (side_len_x - 1 - 2 * i, -side_len_y + 1 + 2 * j),
                    (-side_len_x + 1 + 2 * i, -side_len_y + 1 + 2 * j)
                ])
        excluded_points = np.array(excluded_points)

        mask = np.ones(len(points), dtype=bool)
        for excluded_point in excluded_points:
            mask &= ~np.all(points == excluded_point, axis=1)
        points = points[mask]

    return points, excluded_points

# Fonction pour ajouter du bruit et sauvegarder les résultats
def plot_qam_constellation_with_noise(M, snr_db):
    exclusions = {32: 4, 128: 16, 512: 64, 2048: 196}
    exclude_points = exclusions.get(M, 0)

    points, excluded_points = generate_qam_constellation(M, exclude_points)

    complex_points = points[:, 0] + 1j * points[:, 1]

    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(complex_points) ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*complex_points.shape) + 1j * np.random.randn(*complex_points.shape))
    noisy_points = complex_points + noise

    noisy_points = np.column_stack((noisy_points.real, noisy_points.imag))

    # Calcul des composantes I et Q
    I = noisy_points[:, 0]
    Q = noisy_points[:, 1]

    # Calcul de l'énergie des symboles
    energies = np.abs(complex_points) ** 2
    energies = np.abs(complex_points) ** 2
    # Calcul de la phase des symboles
    phase = np.angle(complex_points)

    # Sauvegarde des résultats dans un fichier Excel avec coordonnées comme index
    df = pd.DataFrame({
        'I (In-Phase)': points[:, 0],
        'Q (Quadrature)': points[:, 1],
        'Energy': energies,
        'Phase (radians)': phase
    })

    desktop_path = r'C:\Users\USER\Desktop\M-QAM'
    excel_filename = os.path.join(desktop_path, f'{M}-QAM_constellation_with_noise_SNR_{snr_db}.xlsx')
    df.to_excel(excel_filename, index=False)

    # Affichage du diagramme de constellation
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Original Symbols')
    plt.scatter(noisy_points[:, 0], noisy_points[:, 1], color='orange', alpha=0.7, label='Noisy Symbols')

    if len(excluded_points) > 0:
        plt.scatter(excluded_points[:, 0], excluded_points[:, 1], color='red', marker='x', label='Excluded Symbols')

    plt.grid(True)
    plt.title(f'{M}-QAM Constellation Diagram with Noise (SNR = {snr_db} dB)')
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    max_val = np.max(np.abs(points)) + 2
    ticks = np.arange(-max_val, max_val + 1, 2)
    plt.xticks(ticks[ticks != 0])
    plt.yticks(ticks[ticks != 0])

    plt.axhline(0, color='black', linewidth=1.2)
    plt.axvline(0, color='black', linewidth=1.2)

    pdf_filename = os.path.join(desktop_path, f'{M}-QAM_constellation_with_noise_SNR_{snr_db}.pdf')
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig()
    plt.close()

    print(f'Noisy constellation diagram saved at: {pdf_filename}')
    print(f'Excel file saved at: {excel_filename}')

# Entrée utilisateur
M = int(input("Enter the size of QAM (4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096): "))
valid_sizes = [4, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096]
if M in valid_sizes:
    snr_db = float(input("Enter the Signal-to-Noise Ratio (SNR) in dB: "))
    print("Generating noisy QAM constellation and exporting to PDF and Excel...")
    plot_qam_constellation_with_noise(M, snr_db)
    print("PDF and Excel files generated successfully with noise.")
else:
    print(f"Invalid QAM size. Please choose from: {valid_sizes}.")