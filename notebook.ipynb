{
  "metadata": {
    "kernelspec": {
      "name": "python",
      "display_name": "Python (Pyodide)",
      "language": "python"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "python",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8"
    }
  },
  "nbformat_minor": 4,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "source": "<h1>Predikcija cene automobila</h1>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h2>Uvod</h2>\n",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Tema kojom ćemo se baviti jeste regresija, odnosno kako da predvidimo vrednost promenljive y na osnovu jedne ili više nezavnisnih promenljivih.\nKada imamo jednu nezavisnu promenljivu reč je o jednostrukoj regresiji, mi se nećemo time baviti, već ćemo raditi sa višestrukom regresijom gde imamo više nezavisnih varijabli.\n</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>U ovom projektu pokušaćemo da što preciznije predvidimo cenu nekog automobila na osnovu njegovoih podataka. Koristićemo više modela različitih regresija kako bismo zaključili koji je najbolji model.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h2>Dataset</h2>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Dataset nam predstavlja skup podataka. Postoje dva skupa, trening i test. Ideja jeste da svaki model modelujemo što bolje nad \npodacima iz trening skupa, jer su to podaci koji su nam poznati. Dok ćemo podatke iz test skupa koristiti kako bismo odredili koliko nam je\nodređeni model dobar, odnosno koliko je naša pretpostavka tačna.</p>\n<p>Zadati dataset sadrži sledeće nezavisne promenljive: <br> <table>\n    <tr><th>brand</th></tr>\n    <tr><th>model</th></tr>\n    <tr><th>model_year</th></tr>\n    <tr><th>milage</th></tr>\n    <tr><th>fuel_type</th></tr>\n    <tr><th>transmission</th></tr>\n    <tr><th>ext_col</th></tr>\n    <tr><th>int_col</th></tr>\n    <tr><th>accident</th></tr>\n    <tr><th>engine</th></tr>\n</table>\nI zavisnu promenljivu što je cena automobila</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h2>Metrika</h2>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Da bismo odredili tačnost nekog modela moramo imati neku meru. U našem modelu kao meru ćemo uzeti prilagodjeni $r^2$, naravno pored ove\nmere postoji još mera kao što je MSE(Mean squared error). Naša mera je prilagođeni koeficijent determinacije. Koeficijent determinacije odnosno $r^2$ ima vrednost u opsegu [0,1]. Što smo bliži jedinici mera je bolja, odnosno model je bolji. Po pravilu $r^2$ nikada ne opada, što nije dobro, jer kada bismo dodavali stalno nezavisne promenljive on bi uvek ostao isti ili bi se povećavao što dovodi do pogrešnom zaključka. Stoga uzimamo prilagođeni koeficijent determinacije jer je on modifikacija $r^2$ i uzima u obzir kompleksnost modela i veličinu podataka.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h2>Metodologija</h2>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Kada smo objasnili koju metriku i koji dataset koristimo, potrebno je napraviti modele. </p>\n<p>Pre svega neophodne ja da test podataka namestimo našim potrebama, u nastavku će biti opsiano kako su se određene kolone transformisale u očekivane vrednosti koje bi modeli trebali da prime. Modeli ne mogu da rukuju stringovima, samo brojevima.</p>\n<h3>Outlier-i</h3>\n<p>Kada smo pripremili podatke sledeći korak jeste uklanjanje outlier-a. Outlieri su vrednosti koje značajno odstupaju od ostatka podataka i mogu dovesti do netačnih rezultata modela. Postoji nekoliko metoda za detekciju i uklanjanje outlier-a u regresiji, mi smo koristili interkvartilni opseg (IQR). Kvartil je statistička mera koja nam predstavlja jednu četvrtinu modela,dok je IQR razlika između trećeg i prvog kvartila podataka. Stoga uklanjamo sve one vrednosti koje su udaljene od IQR-a, što nam predstavlja postupak uklanjanja outlier-a. </p>\n<p>Uklanjanjem outlier-a dobijamo bolje mere modela.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>Modeli</h3>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Modeli koje ćemo trenirati su sledeći\n<table>\n    <tr><th>1. LinearRegression</th></tr>\n    <tr><th>2. RandomForest</th></tr>\n    <tr><th>3. DecisionTree</th></tr>\n    <tr><th>4. KNearestNeighbors</th></tr>\n</table>\nSvaki model drugačije deluje na podatke i ima drugačiji rezultat. Nisu svi modeli pogodni za određeni problem. Naš je cilj odrediti koji je model najbolji za naš problem. U nastavku će biti opisano kako svaki model funkcioniše.\n</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>Hiperparametri</h3>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p> hiperparametri su parametri modela koji se postavljaju pre nego što proces treniranja počne. Ovi parametri utiču na ponašanje samog modela, ali nisu direktno naučeni iz podataka tokom treninga. Stoga potrebno je testirati, odnosno uvrstiti različite vrednosti za hiperparametre i videti koje vrednosti donose najbolju meru za određen model. </p>\n<p>Podešavanje hiperparametara je bitno jer direktno utiču na model, samim tim ako dobro namestimo određene hiperparametre poboljšaćemo meru modela.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h2>Implementacija</h2>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>Nameštanje skupa podatak</h3>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Nezavisne promenljive brand, model, modelYear ne diramo, jer nam odgovaraju njihove vrednosti</p>\n",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Kolonu \"milage\" je string koji sadrži broj i karaktere \"mi.\". Ovu kolonu smo sredili tako što smo od stringa izdvojili broj i pretvorili u objekat tipa float.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Kolona \"fuel_type\" je prva kolona koja ima nedostujuće vrednosti, stoga smo proverili kojih vrednosti ima najviše i time smo popunili prazna polja.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Kolona \"engine\" je najzahtevnija, jer u sebi ima dosta podataka. Razdvojena je na dve kolone \"Horsepower\" i \"Engine_power\". Potrobne je bilo iy glavnog stringa izdvojiti ova dva podataka bez karaktera i pretvoriti ih u objekat tipa float. Na kraju dodamo dve kolone u dataset, dok smo početnu kolonu izbrisali iz skupa podatak, jer nam više neće biti potrebna. Kada smo dobili dve nove kolone imale su nedostujuće vrednosti koje smo popunili srednjom vrednošću</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Kolona \"Transmission\" predstavlja koliko brzina ima menjač, iz ove kolone smo izdvojili broj koji to predstavlja i pretvorili ga u objekat tipa int.</p>\n",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Kolona \"accident\" predstavlja da li je vodjena evidencija o nesrećama. Postoje samo dva slučaja, tačno ukoliko jeste ili netačno ukoliko nije. Neka polja su bila None pa im je dodeljena vrednost bila da nisu izveštavani o nesrećama</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Kolone \"int_col\" i \"ext_col\" su specificne jer nemaju nepopunjena polja, već imaju karakter \"-\", što takođe treba gledati kao nedostujuća vrednost. Stoga su ta polja popunjena sa najzastupljenim bojama kod automobila, u našem slulaju je crna boja.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Ostalo je još da namestimo zavisnu promenljivu y, odnosno kolonu \"price\". Ona je predstavljena kao string koji sadrži broj i karakter \"$\". Izdvojen je broj i pretvoren u tip float.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>DecisionTree model</h3>",
      "metadata": {
        "jp-MarkdownHeadingCollapsed": true
      }
    },
    {
      "cell_type": "markdown",
      "source": "<p>Stablo odlučivanja je algoritam mašinskog učenja koji simulira proces donošenja odluka pomoću niza postavljenih pitanja. Radi na principu binarnog stabla. Koraci algoritma:</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<ol>\n  <li><strong>Izbor predefinisanog uslova >> </strong>Algoritam počinje od korena stabla (prvog čvora) i postavlja pitanje o određenoj karakteristici podataka (npr. \"Da li je neka vrednost veća(manja) od neke definisane\"). Postavljanje pitanja ima za cilj podelu podataka na osnovu određenog kriterijuma.</li>\n  <li><strong>Podela podataka >> </strong>Na osnovu odgovora na postavljeno pitanje, podaci se dele na dva podskupa. Na primer, ako je odgovor \"Da\", podaci se dele na grupu gde je uslov zadovoljen, a ako je odgovor \"Ne\", podaci se dele na grupu gde uslov nije zadovoljen.</li>\n  <li><strong>Rekurzija >> </strong>Postupak se zatim rekurzivno ponavlja za svaki od formiranih podskupova, gde se postavljaju nova pitanja u novim čvorovima. Ovaj proces se nastavlja dok se ne postigne određeni kriterijum zaustavljanja, kao što je maksimalna dubina stabla ili određen broj podataka u listu čvora.</li>\n    <li><strong>Formiranje listova stabla >> </strong>Kada se postigne kriterijum zaustavljanja, čvorovi se pretvaraju u listove koji predstavljaju konačne odluke ili prognoze.</li>\n</ol>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p><strong>Rezultat >> </strong>DecisionTreeRegressor(max_depth=20), ima preciznost od: 66.87218208814137%.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>RandomForest model</h3>",
      "metadata": {
        "jp-MarkdownHeadingCollapsed": true
      }
    },
    {
      "cell_type": "markdown",
      "source": "<p>Random Forest je ansambl algoritam mašinskog učenja koji se koristi kako za klasifikaciju tako i za regresiju. Ansambl algoritmi kombinuju više modela kako bi poboljšali ukupne performanse u odnosu na pojedinačne modele. Koraci algoritma:</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<ol>\n    <li><strong>Pristup kombinovanjem >> </strong>Random Forest kombinuje više stabala odlučivanja, koje smo napomenuli kako rade, kako bi stvorio jači model. Svako stablo se trenira na drugačijem podskupu podataka (uzoraka) i/ili na drugačijem podskupu karakteristika (atributa).</li>\n    <li><strong>Slučajan odabir podataka i atributa >> </strong>Prilikom treniranja svakog stabla, Random Forest nasumično izabire podskup podataka sa zamenskim uzorcima (bootstraping). Takođe, za svaki čvor u svakom stablu, nasumično se biraju određeni atributi za razmatranje pri donošenju odluke.</li>\n    <li><strong>Glasanje, izbor rešenja >> </strong>Kada se pravi predikcija, svako stablo daje svoj vlastiti izlaz. Konačna predikcija Random Foresta obično se određuje glasanjem, tj. izborom izlaza koja je većinom predviđena od strane pojedinačnih stabala.</li>\n    <li>Treba napomenuti da je ovaj algoritam dobro otporan na outlier-e</li>\n</ol>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p><strong>Rezultat >> </strong>RandomForestRegressor(n_estimators=50, random_state=42), ima preciznost od: 71.33751063815986 %.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>LinearRegression model</h3>",
      "metadata": {
        "jp-MarkdownHeadingCollapsed": true
      }
    },
    {
      "cell_type": "markdown",
      "source": "<p>Linearna regresija je statistički model koji se koristi u mašinskom učenju za analizu odnosa između zavisne promenljive (\"price\") i više nezavisnih promenljivih (atributa), u našem slučaju. Osnovna ideja linearne regresije je predstaviti odnos između promenljivih kao linearnu funkciju.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "Model linearne regresije ima oblik:\ny = b0 + b1*x1 + b2*x2 + ... + bn*xn∗x \r\n2\r\n​\r\n +…+b \r\nn\r\n​\r\n ∗x \r\nn\r\n​\r\n \r\n\r\ngde je:",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<ul>\n    <li>y zavisna promenljiva</li>\n    <li>b0 je intercept (pomak) modela, tj. vrednost y kada su sve nezavisne promenljive(Xn) jednake 0.</li>\n     <li>b1, b2, ... bn su koeficijenti koji stoje uz nezavisne promenljive</li>\n    <li>x1, x2, ... xn su nezavisne promenljive</li>\n</ul>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Cilj linearne regresije jeste da pronađe b1, b2, ... bn tako da zbir kvadratnih grešska između i-te stvarne vrednosti i i-te predviđenne vrednosti bude što manja, jer što je manja greška to je naš model precizniji.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p><strong>Rezultat >> </strong>LinearRegression(), ima preciznost od: 58.56394516155683 %.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>KNearestNeighbors model</h3>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>K-Nearest Neighbors (KNN) je algoritam mašinskog učenja koji se koristi za klasifikaciju i regresiju. Osnovna ideja KNN-a je da predviđanja za nove podatke pravimo na osnovu sličnosti tih podataka sa već poznatim podacima u trening skupu. Opis algoritma:</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<ol>\n    <li><strong>Klasifikacija >> </strong>Za svaki novi podatak, algoritam identifikuje k najbližih suseda u trening skupu, koristeći neku meru udaljenosti (npr. euklidska udaljenost).</li>\n    <li><strong>Regresija >> </strong>Za regresiju uzima se srednja vrednost numeričkih vrednosti ciljne promenljive među k najbližih suseda.</li>\n    <li><strong>Izbor hiperparametra k >> </strong>Ovaj parametar predstavlja broj suseda koji se gledaju prilikom predvidjanja algoritma.</li>\n</ol>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p><strong>Rezultat >> </strong>KNeighborsRegressor(n_neighbors=5), ima preciznost od: 60.587007316127625 %.</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h3>Podešavanje hiperparametara</h3>\n<p>Kada smo napravili modele sa podrazumevanim parametrima, potrebno je da ih testiramo sa drugačijim vrednostima i vidimo kako se model ponaša. Implementirana funkcija čita sve parametre za svaki model iz .json fajla i vraća najbolje istreniran model sa parametrom koji mu je prosleđen. \nNavedeni su hiperparametri koji su promenljivi za modele:\n<table>\n    <tr><th>DecisionTree</th>\n    <td>Maksimalna dubina stabla</td></tr>\n    <tr><th>RandomForest</th>\n    <td>Broj stabala</td></tr>\n    <tr><th>KNearestNeighbors</th>\n    <td>Broj k</td></tr>\n    \n    \n</table>\n</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<h2>Zaključak</h2>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Svaki od modela sa podešenim hiperparametrom tako da im prilagođeni r^2 bude najveći mogući smo smestili u hešmapu, gde je ključ model, vrednost njegova mera, odnosno prilagođeni r^2. Upoređivanjem mere r^2 dobili smo najbolji model za predikciju cene automobila. </p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p><strong>Zaključak >> </strong>Najbolji model koji koristimo za ovaj set podataka za predikciju cene automobila jeste: RandomForestRegressor(n_estimators=50, random_state=42)</p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>Ovo nisu svi modeli preko kojih možemo da predvidimo cenu automobila, ovo su samo neki od najpoznatijih modela. Prema rezultatu možemo videti da postoje male oscilacije od 10% preciznosti među modelima što nam govori koji model bi bilo bolje da koristimo za ovakve probleme. Naravno ako promenimo problem ne mora da znači da će i modeli koje koristimo biti isti.  </p>",
      "metadata": {}
    },
    {
      "cell_type": "markdown",
      "source": "<p>U daljem pronalasku još preciznijeg rešenja možemo dodati druge modele, koristiti dodatne hiperparametre za svaki model kao i njihove kombinacije. Uvek možemo optimizovati problem i prilagoditi ga određenim uslovima.</p>",
      "metadata": {}
    },
    {
      "cell_type": "code",
      "source": "",
      "metadata": {
        "trusted": true
      },
      "outputs": [],
      "execution_count": null
    }
  ]
}