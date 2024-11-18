To use this setup:

Create all the files above in your working directory
Install required packages if not already installed:

Modules needed
pip install pandas numpy matplotlib seaborn scikit-learn

Run analysis with
python main.py --file your_data.csv --target-column YourColumn

For specific analysis
python main.py --file your_data.csv \
               --date-column DateColumn \
               --id-column IDColumn \
               --target-column TargetColumn \
               --numerical-columns Column1 Column2 Column3


The script will:

Create a timestamped results directory
Run all analyses (correlations  outliers  isolation forest)
Save results to an Excel file in the results directory
Generate correlation heatmap visualizations


python main.py --file your_data.csv \
               --date-column DateColumn \
               --id-column IDColumn \
               --target-column TargetColumn \
               --numerical-columns Column1 Column2 Column3
			   
python main.py --file your_data.csv --date-column DateColumn --id-column IDColumn --target-column TargetColumn --numerical-columns Column1 Column2 Column3
python main.py --file swaption_data.csv --date-column ValuationDate --id-column TradeId --target-column Theta --numerical-columns IRVega IRDelta DaysToExpiry DaysToNextBusinessDay
			   
			   IRVega IRDelta DaysToExpiry DaysToNextBusinessDay

C:\Users\syrio\OneDrive\Documents\ai_use\Proper_project_split