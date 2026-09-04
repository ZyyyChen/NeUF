<<<<<<< ours
import xlwings as xw
import os
import sys
from PySide6.QtWidgets import QApplication, QProgressDialog
from PySide6.QtCore import Qt

def check_ref_patients_echo_v3(main_dir):
    excel_path = os.path.join(main_dir, 'Ref_Files', 'RefPatientsUS.xlsx')
    app = xw.App(visible=False, add_book=False)

    # ---- Qt app (ne crée rien si déjà existante) ----
    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication(sys.argv)

    try:
        wb = app.books.open(excel_path)
        ws_ref = wb.sheets['Ref_patients']
        ws_builder = wb.sheets['Ref_builder']

        # Using used_range to be safe
        last_row_ref = ws_ref.used_range.last_cell.row
        ref_patients_echo = []

        def clean_val(v):
            if v is None: return ""
            if isinstance(v, (int, float)):
                if isinstance(v, float) and v.is_integer():
                    return str(int(v))
                return str(v)
            return str(v).strip()

        print(f"\n--- Checking Ref_patients (Rows 3 to {last_row_ref}) ---")

        # ---- Progress dialog (remplace tqdm) ----
        progress = QProgressDialog(
            "Checking Ref_patients...",
            None,                # pas de bouton cancel → même comportement que tqdm
            3,
            last_row_ref
        )
        progress.setWindowTitle("Processing Excel")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        for row_idx in range(3, last_row_ref + 1):
            progress.setValue(row_idx)
            QApplication.processEvents()

            index_val = ws_ref.range(f'A{row_idx}').value
            name_in_c = ws_ref.range(f'C{row_idx}').value  # The formula/result cell

            resolved_name = "__"

            if index_val is not None:
                try:
                    b_idx = int(index_val)
                    v_a = ws_builder.range(f'A{b_idx}').value
                    v_b = ws_builder.range(f'B{b_idx}').value
                    v_c = ws_builder.range(f'C{b_idx}').value

                    if v_a or v_b or v_c:
                        resolved_name = f"{clean_val(v_a)}_{clean_val(v_b)}_{clean_val(v_c)}"

                except Exception as e:
                    print(f"Row {row_idx}: Error accessing Builder row {index_val} - {e}")

            # Fallback to Column C if builder resolution failed
            if resolved_name == "__" and name_in_c:
                resolved_name = clean_val(name_in_c)

            if resolved_name != "__" and resolved_name != "":
                row_data = ws_ref.range(f'D{row_idx}:G{row_idx}').value
                row_values = [name_in_c] + [
                    v if v is not None else "" for v in row_data
                ]
                ref_patients_echo.append(row_values)

        progress.close()

        # --- Processing ReconstructionParameters ---
        ws_params = wb.sheets['ReconstructionParameters']
        last_row_params = ws_params.used_range.last_cell.row
        params_patients_echo_v2 = []

        print(f"--- Checking Parameters (Rows 3 to {last_row_params}) ---")
        for row_idx in range(3, last_row_params + 1):
            #raw_vals = ws_params.range(f'C{row_idx}:P{row_idx}').value
            raw_vals = ws_params.range(f'C{row_idx}:I{row_idx}').value
            cleaned_params = []
            for v in raw_vals:
                try:
                    cleaned_params.append(float(v) if v is not None else 0.0)
                except:
                    cleaned_params.append(0.0)
            params_patients_echo_v2.append(cleaned_params)

        print(f"\nFinal Count: Ref={len(ref_patients_echo)}, Params={len(params_patients_echo_v2)}")
        
        return ref_patients_echo, params_patients_echo_v2

    finally:
        if 'wb' in locals():
            wb.close()
        app.quit()

 
def check_ref_patients_update(main_dir, idx):
    excel_path = os.path.join(main_dir, 'Ref_Files', 'RefPatientsUS.xlsx')
    app = xw.App(visible=False, add_book=False)
    wb = app.books.open(excel_path)
    # --- Processing ReconstructionParameters ---
    ws_params = wb.sheets['ReconstructionParameters']
    last_row_params = ws_params.used_range.last_cell.row
    params_patients_echo_v2 = []
    raw_vals = ws_params.range(f'C{idx+3}:I{idx+3}').value
    cleaned_params = []
    for v in raw_vals:
        try:
            cleaned_params.append(float(v) if v is not None else 0.0)
        except:
            cleaned_params.append(0.0)
    params_patients_echo_v2.append(cleaned_params)
    
    
    # It is vital to close the app, otherwise the file stays locked
    wb.close()
    app.quit()
    return params_patients_echo_v2
=======
import xlwings as xw
import os
import sys
from PySide6.QtWidgets import QApplication, QProgressDialog
from PySide6.QtCore import Qt

def check_ref_patients_echo_v3(main_dir):
    excel_path = os.path.join(main_dir, 'Ref_Files', 'RefPatientsUS.xlsx')
    app = xw.App(visible=False, add_book=False)

    # ---- Qt app (ne crée rien si déjà existante) ----
    qt_app = QApplication.instance()
    if qt_app is None:
        qt_app = QApplication(sys.argv)

    try:
        wb = app.books.open(excel_path)
        ws_ref = wb.sheets['Ref_patients']
        ws_builder = wb.sheets['Ref_builder']

        # Using used_range to be safe
        last_row_ref = ws_ref.used_range.last_cell.row
        ref_patients_echo = []

        def clean_val(v):
            if v is None: return ""
            if isinstance(v, (int, float)):
                if isinstance(v, float) and v.is_integer():
                    return str(int(v))
                return str(v)
            return str(v).strip()

        print(f"\n--- Checking Ref_patients (Rows 3 to {last_row_ref}) ---")

        # ---- Progress dialog (remplace tqdm) ----
        progress = QProgressDialog(
            "Checking Ref_patients...",
            None,                # pas de bouton cancel → même comportement que tqdm
            3,
            last_row_ref
        )
        progress.setWindowTitle("Processing Excel")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        for row_idx in range(3, last_row_ref + 1):
            progress.setValue(row_idx)
            QApplication.processEvents()

            index_val = ws_ref.range(f'A{row_idx}').value
            name_in_c = ws_ref.range(f'C{row_idx}').value  # The formula/result cell

            resolved_name = "__"

            if index_val is not None:
                try:
                    b_idx = int(index_val)
                    v_a = ws_builder.range(f'A{b_idx}').value
                    v_b = ws_builder.range(f'B{b_idx}').value
                    v_c = ws_builder.range(f'C{b_idx}').value

                    if v_a or v_b or v_c:
                        resolved_name = f"{clean_val(v_a)}_{clean_val(v_b)}_{clean_val(v_c)}"

                except Exception as e:
                    print(f"Row {row_idx}: Error accessing Builder row {index_val} - {e}")

            # Fallback to Column C if builder resolution failed
            if resolved_name == "__" and name_in_c:
                resolved_name = clean_val(name_in_c)

            if resolved_name != "__" and resolved_name != "":
                row_data = ws_ref.range(f'D{row_idx}:G{row_idx}').value
                row_values = [name_in_c] + [
                    v if v is not None else "" for v in row_data
                ]
                ref_patients_echo.append(row_values)

        progress.close()

        # --- Processing ReconstructionParameters ---
        ws_params = wb.sheets['ReconstructionParameters']
        last_row_params = ws_params.used_range.last_cell.row
        params_patients_echo_v2 = []

        print(f"--- Checking Parameters (Rows 3 to {last_row_params}) ---")
        for row_idx in range(3, last_row_params + 1):
            #raw_vals = ws_params.range(f'C{row_idx}:P{row_idx}').value
            raw_vals = ws_params.range(f'C{row_idx}:I{row_idx}').value
            cleaned_params = []
            for v in raw_vals:
                try:
                    cleaned_params.append(float(v) if v is not None else 0.0)
                except:
                    cleaned_params.append(0.0)
            params_patients_echo_v2.append(cleaned_params)

        print(f"\nFinal Count: Ref={len(ref_patients_echo)}, Params={len(params_patients_echo_v2)}")
        
        return ref_patients_echo, params_patients_echo_v2

    finally:
        if 'wb' in locals():
            wb.close()
        app.quit()

 
def check_ref_patients_update(main_dir, idx):
    excel_path = os.path.join(main_dir, 'Ref_Files', 'RefPatientsUS.xlsx')
    app = xw.App(visible=False, add_book=False)
    wb = app.books.open(excel_path)
    # --- Processing ReconstructionParameters ---
    ws_params = wb.sheets['ReconstructionParameters']
    last_row_params = ws_params.used_range.last_cell.row
    params_patients_echo_v2 = []
    raw_vals = ws_params.range(f'C{idx+3}:I{idx+3}').value
    cleaned_params = []
    for v in raw_vals:
        try:
            cleaned_params.append(float(v) if v is not None else 0.0)
        except:
            cleaned_params.append(0.0)
    params_patients_echo_v2.append(cleaned_params)
    
    
    # It is vital to close the app, otherwise the file stays locked
    wb.close()
    app.quit()
    return params_patients_echo_v2
>>>>>>> theirs
