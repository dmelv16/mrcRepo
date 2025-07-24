import subprocess
import datetime
import traceback
# Get current time
now = datetime.datetime.now()
early_cutoff = now.replace(hour=7, minute=30, second=0, microsecond=0)

# List of scripts to run
always_run_scripts = [
    
    r"C:\Users\DMelv\Documents\code\scrapeBR.py",
    r"C:\Users\DMelv\Documents\code\removeHeaders.py",
    # r"C:\Users\DMelv\Documents\code\modelv3ScrapedText.py",
    r"C:\Users\DMelv\Documents\code\modelv4Scrape.py",
    r"C:\Users\DMelv\Documents\code\oddsAPIbs.py",
    r"C:\Users\DMelv\Documents\code\scrapeMLB.py",
    r"C:\Users\DMelv\Documents\code\statcastV2.py",
    r"C:\Users\DMelv\Documents\code\rosterAPIscrape.py",
    r"C:\Users\DMelv\Documents\code\apiBattingOrderv2.py",
    r"C:\Users\DMelv\Documents\code\sqlMLBqueryv5.py",
    r"C:\Users\DMelv\Documents\code\addColumnsv2.py",
    # r"C:\Users\DMelv\Documents\code\mergeRuns.py",
    # r"C:\Users\DMelv\Documents\code\mlbTrainv2.py",    
    # r"C:\Users\DMelv\Documents\code\mlbFeaturebuildv3.py",
    # r"C:\Users\DMelv\Documents\code\aiFinalReportMLBv1.py",
    r"C:\Users\DMelv\Documents\code\artemisPredictor.py",
    r"C:\Users\DMelv\Documents\code\minervaPredictor.py",
    # r"C:\Users\DMelv\Documents\code\apiCurrentFixtureScrape.py",
    # #r"C:\Users\DMelv\Documents\code\apiStatisticsScrape.py",
    # r"C:\Users\DMelv\Documents\code\apiSoccerIndexes.py",
    # r"C:\Users\DMelv\Documents\code\officialAPIstack.py",
    # r"C:\Users\DMelv\Documents\code\betAlgov7.py",    
    # # r"C:\Users\DMelv\Documents\code\nhlScrape.py",
    # # r"C:\Users\DMelv\Documents\code\nhlModelv1.py",
    # r"C:\Users\DMelv\Documents\code\currentYearSoccerScrape.py",
    # r'C:\Users\DMelv\Documents\code\indexesSoccerUpdated.py',
    # r"C:\Users\DMelv\Documents\code\stackModelSoccer.py",
    # r'C:\Users\DMelv\Documents\code\betAlgov2.py',
    # r"C:\Users\DMelv\Documents\code\cbbScrapev2.py",
    # r"C:\Users\DMelv\Documents\code\cbbIndexes.py",
    # r"C:\Users\DMelv\Documents\code\DMI.py",
    # r"C:\Users\DMelv\Documents\code\cbbModelv1.py",
    #r"C:\Users\DMelv\Documents\code\baseballModelv3.py"
]


# # List of early-only scripts
# early_scripts = [
#     r"C:\Users\DMelv\Documents\code\mlbAPIscrapeV2.py",
#     r"C:\Users\DMelv\Documents\code\statcastV2.py",
#     r"C:\Users\DMelv\Documents\code\mlbAdvancedv2.py",
#     r"C:\Users\DMelv\Documents\code\rosterAPIscrape.py",
#     r"C:\Users\DMelv\Documents\code\apiBattingOrder.py",
#     r"C:\Users\DMelv\Documents\code\mlbPytorchV1.py",
#     r"C:\Users\DMelv\Documents\code\apiScheduler.py"
# ]

def run_script(path):
    try:
        print(f"Running: {path}")
        subprocess.run(["python", path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script failed: {path}")
        print(f"Error: {e}")
    except Exception:
        print(f"Unexpected error in script: {path}")
        traceback.print_exc()

# # Run early scripts
# if now < early_cutoff:
#     for script in early_scripts:
#         run_script(script)
# else:
#     print("Skipping early scripts because it is after 07:30 AM.")

# Run always-run scripts
for script in always_run_scripts:
    run_script(script)
