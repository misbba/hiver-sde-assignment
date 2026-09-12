import os
import pandas as pd

def generate_sample_dataset(output_path: str):
    """
    Generates a realistic sample dataset mirroring Kaggle's Customer Support on Twitter (twcs.csv) format.
    Used for instant reproducibility when full 3M twcs.csv dataset is not present in data/raw/.
    """
    data = [
        # AppleSupport - Software & OS Update Issues
        {"tweet_id": 101, "author_id": "customer_1", "inbound": True, "created_at": "Tue Oct 31 22:10:00 +0000 2017", "text": "@AppleSupport My iPhone screen went completely black after the latest update and won't turn on!", "response_by_tweet_id": "102", "in_response_to_tweet_id": None},
        {"tweet_id": 102, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 22:15:00 +0000 2017", "text": "@customer_1 We're here to help! Connect your iPhone to iTunes/Finder on a Mac/PC and enter Recovery Mode to restore or update your device.", "response_by_tweet_id": None, "in_response_to_tweet_id": "101"},

        # AppleSupport - Hardware & Battery Issues
        {"tweet_id": 103, "author_id": "customer_2", "inbound": True, "created_at": "Tue Oct 31 22:12:00 +0000 2017", "text": "@AppleSupport My iPhone 11 battery drains from 100% to 10% in just two hours. Is this a hardware glitch?", "response_by_tweet_id": "104", "in_response_to_tweet_id": None},
        {"tweet_id": 104, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 22:18:00 +0000 2017", "text": "@customer_2 Battery health is vital. Check Settings > Battery > Battery Health, and send us a DM with your iOS version.", "response_by_tweet_id": None, "in_response_to_tweet_id": "103"},

        # AppleSupport - Software & OS Update Issues
        {"tweet_id": 105, "author_id": "customer_3", "inbound": True, "created_at": "Tue Oct 31 22:20:00 +0000 2017", "text": "@AppleSupport Ever since updating to iOS 17, WiFi keeps disconnecting every few minutes.", "response_by_tweet_id": "106", "in_response_to_tweet_id": None},
        {"tweet_id": 106, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 22:25:00 +0000 2017", "text": "@customer_3 Let's fix that. Go to Settings > General > Transfer or Reset > Reset Network Settings.", "response_by_tweet_id": None, "in_response_to_tweet_id": "105"},

        {"tweet_id": 107, "author_id": "customer_4", "inbound": True, "created_at": "Tue Oct 31 22:30:00 +0000 2017", "text": "@AppleSupport My phone is stuck on the Apple logo during software update. What should I do?", "response_by_tweet_id": "108", "in_response_to_tweet_id": None},
        {"tweet_id": 108, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 22:35:00 +0000 2017", "text": "@customer_4 Connect to Finder/iTunes on a Mac/PC and enter Recovery Mode to restore or update your device.", "response_by_tweet_id": None, "in_response_to_tweet_id": "107"},

        # AppleSupport - Account & iCloud Access
        {"tweet_id": 109, "author_id": "customer_5", "inbound": True, "created_at": "Tue Oct 31 22:40:00 +0000 2017", "text": "@AppleSupport Forgot my Apple ID password and security questions. Cannot sign into iCloud.", "response_by_tweet_id": "110", "in_response_to_tweet_id": None},
        {"tweet_id": 110, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 22:45:00 +0000 2017", "text": "@customer_5 You can reset your password securely at iforgot.apple.com or via the Support app on a trusted device.", "response_by_tweet_id": None, "in_response_to_tweet_id": "109"},

        {"tweet_id": 111, "author_id": "customer_6", "inbound": True, "created_at": "Tue Oct 31 22:50:00 +0000 2017", "text": "@AppleSupport My iCloud storage says full even after deleting hundreds of photos!", "response_by_tweet_id": "112", "in_response_to_tweet_id": None},
        {"tweet_id": 112, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 22:55:00 +0000 2017", "text": "@customer_6 Photos stay in 'Recently Deleted' for 30 days. Be sure to empty that folder in the Photos app.", "response_by_tweet_id": None, "in_response_to_tweet_id": "111"},

        # AppleSupport - Billing, Subscriptions & Refunds
        {"tweet_id": 113, "author_id": "customer_7", "inbound": True, "created_at": "Tue Oct 31 23:00:00 +0000 2017", "text": "@AppleSupport I was double charged for my Apple Music subscription this month! I want a refund.", "response_by_tweet_id": "114", "in_response_to_tweet_id": None},
        {"tweet_id": 114, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 23:05:00 +0000 2017", "text": "@customer_7 We can help review charges. Please visit reportaproblem.apple.com to check purchase history and request a refund.", "response_by_tweet_id": None, "in_response_to_tweet_id": "113"},

        {"tweet_id": 115, "author_id": "customer_8", "inbound": True, "created_at": "Tue Oct 31 23:10:00 +0000 2017", "text": "@AppleSupport How do I cancel an app subscription that renewed automatically today?", "response_by_tweet_id": "116", "in_response_to_tweet_id": None},
        {"tweet_id": 116, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 23:15:00 +0000 2017", "text": "@customer_8 Go to Settings > [Your Name] > Subscriptions, tap the subscription, and select Cancel Subscription.", "response_by_tweet_id": None, "in_response_to_tweet_id": "115"},

        # AppleSupport - Accessory & Audio Connection
        {"tweet_id": 117, "author_id": "customer_9", "inbound": True, "created_at": "Tue Oct 31 23:20:00 +0000 2017", "text": "@AppleSupport Left AirPod Pro won't connect or charge in the case.", "response_by_tweet_id": "118", "in_response_to_tweet_id": None},
        {"tweet_id": 118, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 23:25:00 +0000 2017", "text": "@customer_9 Clean the charging contacts inside the case and AirPod stem, then hold the setup button on case to reset.", "response_by_tweet_id": None, "in_response_to_tweet_id": "117"},

        # AppleSupport - Store & Repairs
        {"tweet_id": 119, "author_id": "customer_10", "inbound": True, "created_at": "Tue Oct 31 23:30:00 +0000 2017", "text": "@AppleSupport Can I book a Genius Bar appointment for screen replacement online?", "response_by_tweet_id": "120", "in_response_to_tweet_id": None},
        {"tweet_id": 120, "author_id": "AppleSupport", "inbound": False, "created_at": "Tue Oct 31 23:35:00 +0000 2017", "text": "@customer_10 Yes! You can schedule an appointment via support.apple.com or the Apple Support iOS app.", "response_by_tweet_id": None, "in_response_to_tweet_id": "119"},
    ]
    
    # Expand dataset programmatically with clean variants
    expanded = []
    id_counter = 200
    for i in range(25):
        for item in data:
            if item["inbound"]:
                cust_item = dict(item)
                cust_item["tweet_id"] = id_counter
                cust_item["author_id"] = f"user_{id_counter}"
                cust_item["response_by_tweet_id"] = str(id_counter + 1)
                expanded.append(cust_item)
                
                resp_item = dict(data[data.index(item) + 1])
                resp_item["tweet_id"] = id_counter + 1
                resp_item["in_response_to_tweet_id"] = str(id_counter)
                resp_item["text"] = resp_item["text"].replace("@customer_1", f"@user_{id_counter}").replace("@customer_2", f"@user_{id_counter}")
                expanded.append(resp_item)
                id_counter += 2

    df = pd.DataFrame(expanded)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sample dataset successfully created at {output_path} with {len(df)} rows.")
    return df
