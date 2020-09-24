# Faster_R-CNN

# Introduction

Στην παρούσα εργασία θα ασχοληθούμε με τον τομέα της ανίχνευσης αντικειμένων. Πιο συγκεκριμένα θα υλοποιήσουμε σε Keras ένα Faster R-CNN το οποίο και θα εκπαιδεύσουμε ώστε να ανιχνεύει σε μια εικόνα **ανθρώπους**, **σκύλους**, **γάτες**.

Επέλεξα  να τρέξω το μοντέλο στο colab. Ώστε να εκμαιταλευτω την Tesla K80 που παρέχει η google.
Το μοντέλο εκπαιδεύτηκε με 2,412 εικόνες οι οποίες περιείχαν συνολικά 5937 αντικείμενα τα οποία άνηκαν σε μια από τις 3 κατηγορίες. Συνολικά η εκπαίδευση διήρκησε 16 ώρες και το Total loss ήταν 0,79 .

Το implementation για το Faster R-CNN που χρησιμοποίησα είναι το παρακάτω https://github.com/kbardool/keras-frcnn

# Steps

## Prepare dataset

1. Το πρώτο βήμα είναι να βρούμε τα δεδομένα τα οποία θα χρησιμοποιήσουμε. Χρειαζόμαστε εικόνες οι οποίες περιέχουν αντικείμενα από τις 3 κατηγορίες που επιλέξαμε καθώς και τις συντεταγμένες στις οποίες βρίσκονται τα αντικείμενα.
Δεδομένα με αυτές τις πληροφορίες μπορούμε να βρούμε στο **Google’s Open Images Dataset** [link](https://storage.googleapis.com/openimages/web/download.html)

2. Όλα τα βήματα για το πως θα κατεβάσουμε τα δεδομένα μας αλλά και πως θα τα διαμορφώσουμε για την εκπαίδευση και το testing περιγράφονται στο αρχείο [DataDownload.ipynb](https://github.com/ChriBoLab/Faster_R-CNN/blob/master/Code/PrepareDataset/DataDownload.ipynb)

3. Aφού έχουμε κατεβάσει τα δεδομένα μας και τα έχουμε χωρίσει σε φακέλους **train** 80% και **test** 20%, για να είμαστε σε θέση να τρέξουμε το μοντέλο μας είναι απαραίτητο να φτιάξουμε ένα txt αρχείο ονόματι annotations.txt που θα περιγράφει για την κάθε εικόνα που βρίσκεται στον φάκελο train-set τις συντεταγμένες όπου βρίσκονται τα αντικείμενα της. Η παραπάνω διαδικασία υλοποιείται στο αρχείο [PrepareDataForTheModel.ipynb](https://github.com/ChriBoLab/Faster_R-CNN/blob/master/Code/PrepareDataset/PrepareDataForTheModel.ipynb)

## Train

4. Τώρα είμαστε σε θέση να εκπαιδεύσουμε το μοντέλο μας. Τα βήματα για το πως θα τρέξει το μοντέλο περιγράφονται στο παρακάτω σύνδεσμο https://colab.research.google.com/drive/1bfaEGIC4P8Y9NqVIG5Ww2OcHHABWj5fx?usp=sharing

## Testing

5. Αφού εκπαιδευτεί το μοντέλο ήρθε η στιγμή να το τεστάρουμε σε εικόνες που δεν έχουμε ξαναδεί. Πριν ξεκινήσουμε το testing θα πρέπει να επέμβουμε σε ένα αρχείο από το implementation ώστε να έχουμε μια επιπλέον λειτουργικότητα.

Το αρχείο ονομάζεται **test_frcnn.py** και είναι υπεύθυνο για το testing. Όπως θα δούμε στην συνέχεια για να κάνουμε evaluate το μοντέλο θα πρέπει να κρατάμε τις συντεταγμένες από τις εκτιμήσεις τις οποίες έκανε το μοντέλο. Έτσι θα δημιουργήσουμε ένα dataframe στο οποίο θα αποθηκεύουμε όλες τις πληροφορίες από την κάθε εκτίμηση του μοντέλου. Ο επιπλέον κώδικας που θα προσθέσουμε είναι ο παρακάτω:

```python
#to evalueate the model
test_df = pd.DataFrame(columns=['FileName', 'XMin',
                                'XMax', 'YMin', 'YMax', 'ClassName', 'Prob'])

```

```python
#my code
test_df.loc[len(test_df)] = [img_name] + [real_x1] + [real_y1] + [real_x2] + [real_y2] + [key] + [new_probs[jk]]

```
Στα έγγραφα παρέχετε η ανανεωμένη έκδοση αυτού του αρχείου.

6. Αφου εχουμε ανανεωση το αρχειο μας ακολουθουμε τα βηματα απο τον παρακατω συνδεσμο για να κανουμε το testing https://colab.research.google.com/drive/10yZ-lI-fPz9PYYOz-2Gpt6d1Yhwx3tsb?usp=sharing

## Evaluation

7. Έχουμε στα χέρια μας τα αποτελέσματα άρα ήρθε η στιγμή να τα αξιολογήσουμε. Tα βήματα για το πως θα προετοιμάσουμε τα αρχεία για την αξιολόγησή περιγράφονται στα δυο παρακάτω αρχεία **detections.ipynb**, **groundtrouths.ipynb**.

Αφού δημιουργήσαμε τους δυο παραπάνω φακέλους ήρθε η στιγμή να πάρουμε τα αποτελέσματα των μετρικών τα βήματα για το πως θα το κάνουμε αυτό παρουσιάζονται στο αρχείο **RunEvaluation.ipynb**.




# Resutls
### Cats
<table>
  <tr>
    <td></td>
     <td></td>
     <td></td>
  </tr>
  <tr>
    <td><img src="Input/Cat/1.jpg" width=270 height=270></td>
    <td><img src="Input/Cat/5.jpg" width=270 height=270></td>
    <td><img src="Input/Cat/6.jpg" width=270 height=270></td>
  </tr>
 </table>
 
### Dogs
 <table>
  <tr>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><img src="Input/Dog/1.jpg" width=270 height=300></td>
    <td><img src="Input/Dog/2.jpg" width=270 height=300></td>
    <td><img src="Input/Dog/3.jpg" width=270 height=300></td>
  </tr>
 </table>
 
 ### Persons
 <table>
  <tr>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><img src="Input/Person/5.jpg" width=270 height=300></td>
    <td><img src="Input/Person/6.jpg" width=270 height=300></td>
    <td><img src="Input/Person/7.jpg" width=270 height=300></td>
  </tr>
 </table>
 
 # Metrics
 <img src="Input/Evaluation/Cat.png">
  <img src="Input/Evaluation/Dog.png">
   <img src="Input/Evaluation/Person.png">
 
 
 Όπως βλέπουμε και από τις ίδιες τις εικόνες άλλα και από τις μετρικές το μοντέλο είναι πιο κάλο στο να βρίσκει γάτες λιγότερο καλό στο να βρίσκει σκύλους και χειρότερο στο να βρίσκει ανθρώπους. Κάτι τέτοιο μπορούμε να το παρατηρήσουμε και από τις ίδιες τις εικόνες όπου είχαμε πάρα πολλά false positives στους ανθρώπους.
