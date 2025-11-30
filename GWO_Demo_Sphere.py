import numpy as np
import matplotlib.pyplot as plt


# --- 1. ĐỊNH NGHĨA BÀI TOÁN (Sphere Function) ---
# Mục tiêu: Tìm vị trí (x1, x2, ..., x30) sao cho tổng bình phương về 0
def sphere_function(x):
    return np.sum(x ** 2)


# --- 2. THUẬT TOÁN GWO ---
def GWO(obj_func, lb, ub, dim, pop_size, max_iter):
    # Khởi tạo Alpha, Beta, Delta
    Alpha_pos = np.zeros(dim);
    Alpha_score = float("inf")
    Beta_pos = np.zeros(dim);
    Beta_score = float("inf")
    Delta_pos = np.zeros(dim);
    Delta_score = float("inf")

    # Khởi tạo bầy sói ngẫu nhiên
    Positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    convergence = []  # Lưu lịch sử để vẽ biểu đồ

    print("GWO đang chạy...")

    for l in range(0, max_iter):
        for i in range(0, pop_size):
            # Giới hạn không gian (Clip)
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)

            # Tính điểm Fitness
            fitness = obj_func(Positions[i, :])

            # Cập nhật Thủ lĩnh
            if fitness < Alpha_score:
                Alpha_score, Alpha_pos = fitness, Positions[i, :].copy()
            elif fitness < Beta_score:
                Beta_score, Beta_pos = fitness, Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, Positions[i, :].copy()

        convergence.append(Alpha_score)

        # Cập nhật vị trí (Săn mồi)
        a = 2 - l * (2 / max_iter)  # a giảm từ 2 -> 0
        for i in range(0, pop_size):
            for j in range(0, dim):
                # Công thức trung bình cộng 3 con đầu đàn
                r1, r2 = np.random.random(), np.random.random()
                X1 = Alpha_pos[j] - (2 * a * r1 - a) * abs(2 * r2 * Alpha_pos[j] - Positions[i, j])

                r1, r2 = np.random.random(), np.random.random()
                X2 = Beta_pos[j] - (2 * a * r1 - a) * abs(2 * r2 * Beta_pos[j] - Positions[i, j])

                r1, r2 = np.random.random(), np.random.random()
                X3 = Delta_pos[j] - (2 * a * r1 - a) * abs(2 * r2 * Delta_pos[j] - Positions[i, j])

                Positions[i, j] = (X1 + X2 + X3) / 3

    return Alpha_score, convergence


# --- 3. CHẠY DEMO ---
if __name__ == "__main__":
    # Cấu hình bài toán: 30 chiều, 50 con sói, 100 vòng lặp
    best_score, curve = GWO(sphere_function, -100, 100, 30, 50, 100)

    print(f"Kết quả tối ưu (Gần 0 là tốt): {best_score}")

    # Vẽ biểu đồ
    plt.plot(curve, '-r')
    plt.title('Biểu đồ hội tụ GWO trên hàm Sphere')
    plt.xlabel('Vòng lặp');
    plt.ylabel('Fitness')
    plt.show()