import cv2
import pytesseract
import numpy as np

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    return True

def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def print_board(board):
    for row in board:
        print(" ".join(str(num) for num in row))

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Sudoku', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def extract_sudoku_board(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Debug: Show the thresholded image
    cv2.imshow('Thresholded Image', thresh)
    cv2.waitKey(0)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    
    if not contours:
        print("No contours found")
        return None
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Increase precision
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Debug: Check the number of points in approx
        print(f"Number of points in approx: {len(approx)}")
        
        if len(approx) == 4:
            pts = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            ordered_pts = order_points(pts)
            
            # Debug: Draw the points on the image
            debug_image = image.copy()
            for point in ordered_pts:
                cv2.circle(debug_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.imshow('Ordered Points', debug_image)
            cv2.waitKey(0)
            
            dst = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])  # Fixed size for the Sudoku grid
            M = cv2.getPerspectiveTransform(ordered_pts, dst)
            warp = cv2.warpPerspective(image, M, (450, 450))
            
            # Debug: Show the warped image
            cv2.imshow('Warped Image', warp)
            cv2.waitKey(0)
            
            return warp
    
    print("Failed to find a contour with 4 points")
    return None

def preprocess_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Resize the cell to a standard size for better OCR accuracy
    resized = cv2.resize(thresh, (50, 50), interpolation=cv2.INTER_AREA)
    return resized

def ocr_sudoku_board(image):
    board = [[0 for _ in range(9)] for _ in range(9)]
    height, width = image.shape[:2]
    cell_height, cell_width = height // 9, width // 9
    for i in range(9):
        for j in range(9):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            cell = preprocess_cell(cell)
            text = pytesseract.image_to_string(cell, config='--psm 10 -c tessedit_char_whitelist=0123456789')
            try:
                num = int(text.strip())
                board[i][j] = num
            except ValueError:
                board[i][j] = 0
    return board

def main():
    image = capture_image()
    sudoku_image = extract_sudoku_board(image)
    if sudoku_image is not None:
        board = ocr_sudoku_board(sudoku_image)
        print("Captured Sudoku Board:")
        print_board(board)
        if solve_sudoku(board):
            print("Solved Sudoku Board:")
            print_board(board)
        else:
            print("No solution exists")
    else:
        print("Failed to extract Sudoku board from image")

if __name__ == "__main__":
    main()