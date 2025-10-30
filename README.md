# Εξόρυξη Δεδομένων για την Ανάλυση και Πρόβλεψη Κατανάλωσης Ενέργειας σε Έξυπνο Σπίτι

## Οδηγίες για το Dataset

Το αρχείο dataset **ΔΕΝ περιλαμβάνεται στο GitHub repository** λόγω μεγέθους (>100MB). Για να λειτουργήσει η εργασία σου και να τρέξεις τα notebooks, ακολούθησε προσεκτικά τα παρακάτω βήματα:

### 1. Κατέβασε το dataset

- Πήγαινε στο UCI Machine Learning Repository:
  - [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- Εναλλακτικά, κατέβασε απευθείας το ZIP αρχείο:
  - [Direct ZIP Download](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)

### 2. Αποσυμπίεσε το αρχείο

- Κάνε extract το ZIP και εντόπισε το αρχείο `household_power_consumption.txt`.

### 3. Δημιούργησε τον φάκελο `data` (αν δεν υπάρχει ήδη)

```bash
mkdir -p data
```

### 4. Τοποθέτησε το dataset

- Βάλε το αρχείο `household_power_consumption.txt` μέσα στον φάκελο `data` του project σου:
  ```
  energy-consumption-mining/data/household_power_consumption.txt
  ```

---

## Οδηγίες για το περιβάλλον εργασίας (Python venv & βιβλιοθήκες)

### 1. Δημιούργησε και ενεργοποίησε εικονικό περιβάλλον Python

```bash
sudo apt update
sudo apt install python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
```

### 2. Εγκατέστησε τις απαραίτητες βιβλιοθήκες

Αν υπάρχει αρχείο `requirements.txt`:
```bash
pip install -r requirements.txt
```