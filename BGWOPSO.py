import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math


# --- 1. Helper Functions ---

def sigmoid(x, a=10, c=0.5):
    """
    Mô phỏng hàm sigmf : 1 / (1 + exp(-a*(x-c)))
    """
    # Dùng clip để tránh overflow khi tính exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-a * (x - c)))


def load_dat_file(filepath):
    """
    Hàm đọc file M-of-n.dat, xử lý cả các dòng chứa tag
    """
    data_rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Xử lý dòng chứa header
            # Ví dụ: "0 1 0..." -> lấy phần "0 1 0..."
            if line.startswith('['):
                if ']' in line:
                    # Lấy phần sau dấu đóng ngoặc ']'
                    line = line.split(']', 1)[1]
                else:
                    # Nếu dòng chỉ có '[' mà không có ']', bỏ qua để tránh lỗi
                    continue

            # Chuyển chuỗi số còn lại thành list float
            # Sử dụng biến tạm 'nums' để kiểm tra trước khi append
            try:
                nums = [float(x) for x in line.split()]
                if nums:  # Chỉ thêm vào nếu dòng có dữ liệu
                    data_rows.append(nums)
            except ValueError:
                continue  # Bỏ qua nếu dòng chứa ký tự không phải số

    # Chuyển list thành numpy array
    # Lưu ý: Các dòng phải có cùng độ dài (số cột).
    # Nếu file M-of-n.dat có các dòng độ dài khác nhau, numpy sẽ tạo array object (không khuyến khích).
    return np.array(data_rows)

# --- 2. Objective Functions (AccSz & Acc) ---

def calculate_fitness(position, X_train, X_valid, y_train, y_valid):
    """
    Tương đương AccSz.m: Tính fitness dựa trên độ chính xác + số lượng đặc trưng
    """
    # Chuyển vị trí continuous/velocity sang binary (0 hoặc 1)
    selected_features = position > 0.5

    # Nếu không có đặc trưng nào được chọn, trả về vô cùng (phạt nặng)
    if np.sum(selected_features) == 0:
        return float('inf'), 0.0

    # Lọc dữ liệu theo các đặc trưng được chọn
    # position có kích thước bằng số cột feature (bỏ cột cuối là label)
    X_train_subset = X_train[:, selected_features]
    X_valid_subset = X_valid[:, selected_features]

    # KNN Classifier (MATLAB knnclassify mặc định k=1)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_subset, y_train)
    predictions = knn.predict(X_valid_subset)

    accuracy = accuracy_score(y_valid, predictions)

    # Công thức fitness trong bài báo/code:
    # fitness = alpha * error_rate + beta * (num_selected / total_features)
    # Code gốc: SzW = 0.01
    SzW = 0.01
    error_rate = 1 - accuracy
    num_selected = np.sum(selected_features)
    total_features = len(position)

    fitness = (1 - SzW) * error_rate + SzW * (num_selected / total_features)

    return fitness, accuracy


# --- 3. Main Algorithm: BGWOPSO ---

def bgwopso(X, y, search_agents_no=10, max_iter=100):
    """
    Hàm chính thực thi thuật toán lai GWO-PSO
    """
    # Chia dữ liệu train/test (giống phần Demo.m)
    # Lưu ý: Demo.m dùng random permutation fix 50/50. Ở đây dùng train_test_split.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)

    dim = X.shape[1]  # Số lượng features

    # Khởi tạo Alpha, Beta, Delta
    alpha_pos = np.zeros(dim)
    alpha_score = float('inf')

    beta_pos = np.zeros(dim)
    beta_score = float('inf')

    delta_pos = np.zeros(dim)
    delta_score = float('inf')

    # Khởi tạo vị trí ngẫu nhiên (Initialization.m)
    # Positions: (SearchAgents_no, dim)
    positions = np.random.rand(search_agents_no, dim) > 0.5
    positions = positions.astype(float)  # Chuyển về float để tính toán velocity

    # Khởi tạo velocity và convergence curve
    convergence_curve = np.zeros(max_iter)
    velocity = 0.3 * np.random.randn(search_agents_no, dim)

    w = 0.5 + np.random.rand() / 2
    l = 0  # Loop counter

    print(f"Bắt đầu tối ưu hóa với {search_agents_no} agents và {max_iter} vòng lặp...")

    # Main Loop
    while l < max_iter:
        for i in range(search_agents_no):
            # Kiểm tra boundary (nếu cần thiết, code gốc ko check kỹ boundary cho binary)

            # Tính fitness
            fitness, _ = calculate_fitness(positions[i, :], X_train, X_valid, y_train, y_valid)

            # Update Alpha, Beta, Delta
            if fitness < alpha_score:
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()

            if fitness > alpha_score and fitness < beta_score:
                beta_score = fitness
                beta_pos = positions[i, :].copy()

            if fitness > alpha_score and fitness > beta_score and fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        # a giảm tuyến tính từ 2 xuống 0
        a = 2 - l * (2 / max_iter)

        # Cập nhật vị trí các search agents
        for i in range(search_agents_no):
            for j in range(dim):

                # --- Xử lý Alpha ---
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 0.5
                D_alpha = abs(C1 * alpha_pos[j] - w * positions[i, j])

                # Sigmoid transform (eq trong code MATLAB)
                # v1=sigmf(-A1*D_alpha,[10, 0.5]);
                val_sig_alpha = sigmoid(-A1 * D_alpha, 10, 0.5)
                v1_alpha = 1 if val_sig_alpha >= np.random.rand() else 0
                X1 = 1 if (alpha_pos[j] + v1_alpha) >= 1 else 0

                # --- Xử lý Beta ---
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 0.5
                D_beta = abs(C2 * beta_pos[j] - w * positions[i, j])

                val_sig_beta = sigmoid(-A2 * D_beta, 10, 0.5)
                v1_beta = 1 if val_sig_beta >= np.random.rand() else 0
                X2 = 1 if (beta_pos[j] + v1_beta) >= 1 else 0

                # --- Xử lý Delta ---
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 0.5
                D_delta = abs(C3 * delta_pos[j] - w * positions[i, j])

                val_sig_delta = sigmoid(-A3 * D_delta, 10, 0.5)
                v1_delta = 1 if val_sig_delta >= np.random.rand() else 0
                X3 = 1 if (delta_pos[j] + v1_delta) >= 1 else 0

                # --- Cập nhật Velocity (PSO part) ---
                term1 = C1 * np.random.rand() * (X1 - positions[i, j])
                term2 = C2 * np.random.rand() * (X2 - positions[i, j])
                term3 = C3 * np.random.rand() * (X3 - positions[i, j])
                velocity[i, j] = w * (velocity[i, j] + term1 + term2 + term3)

                # --- Cập nhật Position ---
                # Code gốc: xx=sigmf((X1+X2+X3)/3,[10 0.5])+velocity(i,j);
                avg_X = (X1 + X2 + X3) / 3.0
                xx = sigmoid(avg_X, 10, 0.5) + velocity[i, j]

                if xx < np.random.rand():
                    positions[i, j] = 0
                else:
                    positions[i, j] = 1

        convergence_curve[l] = alpha_score
        print(f"Iteration {l + 1}: Best Cost = {alpha_score:.6f}")
        l += 1

    return alpha_score, alpha_pos, convergence_curve


# --- 4. Chạy chương trình (Demo) ---

if __name__ == "__main__":
    # Load dữ liệu từ file M-of-n.dat bạn đã upload
    # Giả định file nằm cùng thư mục, hoặc bạn thay đường dẫn vào đây
    try:
        data = load_dat_file('M-of-n.dat')

        if data.size == 0:
            raise ValueError("File dữ liệu rỗng hoặc không đúng định dạng.")

        # Tách Features (X) và Label (y)
        # Giả định cột cuối cùng là nhãn (Label) giống file covid.dat trong Demo.m
        X = data[:, :-1]
        y = data[:, -1]

        print(f"Dữ liệu loaded: {X.shape[0]} mẫu, {X.shape[1]} đặc trưng.")

        # Cấu hình tham số
        SearchAgents_no = 10
        Max_iteration = 50  # Giảm xuống 50 để test nhanh, code gốc là 100

        # Chạy thuật toán
        best_score, best_pos, curve = bgwopso(X, y, SearchAgents_no, Max_iteration)

        # Tính toán kết quả cuối cùng
        # Tách lại tập train/test giống trong hàm bgwopso để tính Acc lần cuối
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)
        final_fitness, final_acc = calculate_fitness(best_pos, X_train, X_valid, y_train, y_valid)

        selected_indices = np.where(best_pos > 0.5)[0]

        print("\n" + "=" * 40)
        print("KẾT QUẢ TỐI ƯU HÓA")
        print("=" * 40)
        print(f"Best Fitness (Cost): {best_score:.6f}")
        print(f"Final Validation Accuracy: {final_acc * 100:.2f}%")
        print(f"Số lượng đặc trưng đã chọn: {len(selected_indices)}")
        print(f"Indices đặc trưng: {selected_indices}")
        print("=" * 40)

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'M-of-n.dat'. Hãy đảm bảo file nằm cùng thư mục với script này.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")