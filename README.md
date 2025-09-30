# Demografisk Simulator: Kulturell Dynamik (v5.0)

## 1. Översikt

Detta projekt är en **agentbaserad demografisk simulator** för Sverige, byggd i Python med ett webbgränssnitt som drivs av Streamlit. Verktygets primära syfte är att modellera och visualisera de långsiktiga effekterna av demografiska händelser (födslar, dödsfall, migration) och sociala processer (kulturell assimilering) på den svenska befolkningens storlek och sammansättning.

Modellen är deterministisk på makronivå (kalibrerad mot SCB-data) men stokastisk på mikronivå (individuella agenters öden).

### Teknisk Stack
- **Språk:** Python 3.x
- **UI-ramverk:** Streamlit
- **Databehandling:** Pandas, NumPy
- **Visualisering:** Plotly Express

## 2. Installation och Körning

### Förutsättningar
- Python 3.8+
- `pip` och `venv`

### Installationssteg
1.  **Klona projektet:**
    ```bash
    git clone <repository_url>
    cd demografisk_simulator
    ```

2.  **Skapa och aktivera en virtuell miljö:**
    ```bash
    # För Windows
    python -m venv venv
    .\venv\Scripts\activate

    # För macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Installera beroenden:**
    (Skapa en `requirements.txt`-fil med innehållet `streamlit`, `pandas`, `numpy`, `plotly`)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Kör applikationen:**
    Placera `SCB Raw Data.csv` i rotmappen och kör följande kommando:
    ```bash
    streamlit run app.py
    ```

## 3. Projektstruktur

Projektet är uppbyggt modulärt för att separera ansvarsområden (UI, simuleringsmotor, dataladdning).

```
demografisk_simulator/
├── 📂 simulation/              # Simuleringsmotorn (kärnlogik)
│   ├── __init__.py           # Gör mappen till ett Python-paket
│   ├── population.py         # Klass som hanterar agent-populationen och demografiska processer
│   └── engine.py             # Klass som orkestrerar hela simuleringsloopen
│
├── 📂 utils/                  # Hjälpfunktioner och datamoduler
│   ├── __init__.py           # Gör mappen till ett Python-paket
│   ├── data_loader.py        # Funktion för att ladda och förbereda SCB-data
│   ├── parameters.py         # Funktion som definierar alla standardparametrar
│   └── summary.py            # Funktioner för att skapa års-sammanfattningar
│
├── 📜 app.py                  # Huvudfil: Streamlit UI och orkestrering
└── 📜 SCB Raw Data.csv        # Källdata från Statistiska centralbyrån
```

## 4. Arkitektur och Kärnkoncept

### Agentbaserad Modellering (ABM)
Systemet simulerar inte bara aggregerade siffror. Det skapar och hanterar en population av tusentals individuella **agenter**, där varje agent representerar en person. Agenterna lagras i en Pandas DataFrame för effektiv vektoriserad beräkning.

### Kärnkoncept: `swedish_share`
Den centrala variabeln i modellen är `swedish_share`. Det är ett flyttal mellan `0.0` och `1.0` som tilldelas varje agent. Detta värde är en **abstraktion av social och kulturell tillhörighet** till den svenska majoritetskulturen. Det är den primära drivkraften för flera mekanismer:
- **Assimilering:** Värdet kan förändras över tid.
- **Fertilitet:** Påverkar sannolikheten att få barn.
- **Utvandring:** Påverkar sannolikheten att lämna landet.
- **Partner-val:** Påverkar valet av partner vid födsel.

### Klasser och Ansvar
- **`Population` (`simulation/population.py`):** Denna klass är "ägare" till agent-DataFramen. Alla operationer som direkt modifierar populationen (födslar, dödsfall, etc.) är metoder i denna klass. Den är helt frikopplad från Streamlit.
- **`SimulationEngine` (`simulation/engine.py`):** Denna klass är "dirigenten". Den tar emot en `Population`-instans, data och inställningar, och exekverar den årliga simuleringsloopen. Den håller reda på tid, skiljer på historiska år och prognosår, och samlar in resultat.

## 5. Simuleringsflöde (Årlig Cykel)

Simuleringen exekveras i en `for`-loop från `start_year` till `end_year`. För varje år utförs följande steg i strikt ordning:

1.  **Åldrande och Assimilering (`perform_aging_and_assimilation`):**
    - Alla levande agenter blir ett år äldre.
    - Andelen agenter med `swedish_share >= 0.999` beräknas (`P_svensk`).
    - En dynamisk faktor `(P_svensk * 2) - 1` bestämmer riktningen och den relativa hastigheten på assimileringen för året.
    - Förändringen appliceras på alla agenter med `0.0 < swedish_share < 1.0`.

2.  **Dödsfall (`perform_deaths` / `perform_deaths_forecast`):**
    - En åldersbaserad grund-sannolikhet att dö beräknas för varje agent.
    - **Historiska år (1960-2024):** Sannolikheten skalas så att det totala antalet döda i simuleringen exakt matchar värdet i `SCB Raw Data.csv`.
    - **Prognosår (2025+):** Sannolikheten justeras för att gradvis närma sig de målvärden för medellivslängd som angetts i parametrarna.

3.  **Födslar (`perform_births` / `perform_births_forecast`):**
    - En "födselpotential" beräknas för alla fertila kvinnor, baserat på deras `swedish_share` och `fertility_bonuses` (ursprungsregion).
    - **Historiska år:** Potentialen skalas så att totala antalet födslar matchar SCB-data.
    - **Prognosår:** Potentialen skalas så att totala antalet födslar matchar det TFR-värde (Total Fertility Rate) som angetts.
    - Vid födsel ärver barnet `(mor.swedish_share + far.swedish_share) / 2`.

4.  **Migration (`perform_migration` / `perform_migration_forecast`):**
    - **Invandring:** Nya agenter skapas. Antal, ursprung och ålder bestäms av SCB-data (historiskt) eller prognosparametrar (framtid).
    - **Utvandring:** Agenter sätts till `is_alive = False`. Antalet styrs av SCB/prognos. Urvalet viktas av `emigration_bias`, vilket gör att agenter med låg `swedish_share` har högre sannolikhet att väljas.

## 6. Dataflöde och Källor

### Historisk Data
- **Källa:** `SCB Raw Data.csv`.
- **Användning:** Filen agerar som "sanningen" för perioden 1960-2024. Simuleringsmotorn använder den för att kalibrera de årliga händelserna (steg 2, 3, 4 ovan) för att säkerställa att modellens totaler följer den verkliga historiska utvecklingen.

### Standardparametrar
- **Källa:** `utils/parameters.py` och `app.py` (för `forecast_settings`).
- **Ursprung:**
    - **SCB-baserade värden:** Standardvärdena för framtida TFR (`1.59`) och medellivslängd (`85.3`/`87.7` år 2050) är direkt hämtade från **SCB:s befolkningsframskrivning 2024-2070**.
    - **Kalibrerade värden:** Vissa värden är kalibreringar som baseras på observerade trender, men där exakta siffror saknas. Exempel: `fertility_bonuses` är justerade för att TFR-skillnaderna mellan inrikes/utrikes födda ska stämma. `historical_waves` är en förenklad representation av verkliga migrationsmönster.
    - **Rena modellantaganden:** Vissa parametrar är rena antaganden som definierar modellens beteende. Exempel: `base_rate` och `strength_multiplier` för assimilering. Hjälptexterna i UI:t är tydliga med när ett värde är ett modellantagande.

## 7. Framtida Utveckling

Den nuvarande modulära arkitekturen är designad för att enkelt kunna byggas ut. Potentiella framtida förbättringar inkluderar:
- **Geografisk dimension:** Lägga till en `region`-egenskap hos agenter för att simulera urbanisering och regional segregation/integration.
- **Socioekonomisk dimension:** Lägga till egenskaper som `utbildningsnivå` eller `inkomst` för att skapa en mer finkornig modell där dessa faktorer kan påverka demografiska utfall.
- **Nätverksmodellering:** Istället för en global `proportion_swedish_background`, låta en agents assimilering påverkas av dess direkta sociala nätverk.