import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict
from src.config import GOLDEN_SET_PATH, INTENT_TAXONOMY, RANDOM_SEED

np.random.seed(RANDOM_SEED)

# 200 distinct, realistic AppleSupport customer support inquiries
UNIQUE_200_CANDIDATES = [
    # Software & Updates (30 unique examples)
    ("software_update", "AUTO-HANDLE", "@AppleSupport My iPhone screen went completely black after the latest update and won't turn on!", "Connect to computer; enter Recovery Mode via iTunes or Finder to update/restore."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Since updating to iOS 17.2, my WiFi disconnects every 5 minutes continuously.", "Reset Network Settings in Settings > General > Transfer or Reset iPhone."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Phone stuck on black screen with spinning wheel during system update.", "Connect to computer; enter Recovery Mode via iTunes or Finder to update/restore."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Camera app freezes and shows a black view after installing latest update.", "Force quit camera app; restart phone; reset all settings if issue persists."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Can I downgrade from iOS 17 back to iOS 16 without losing my data?", "Downgrading is generally not supported once signing ends; backup before restoring."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Bluetooth disconnects every time I start a phone call post update.", "Forget Bluetooth device and re-pair; reset network settings."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Apps lock up immediately upon launch on iOS 17.1.", "Check App Store for pending app updates; reinstall affected apps."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Keyboard lag is unbearable when typing long messages after update.", "Reset Keyboard Dictionary in Settings > General > Transfer or Reset."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Notification sounds stopped working completely post update.", "Verify Focus modes and Do Not Disturb switches; check Alert volumes."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport iPhone reboots every 10 minutes continuously since updating last night.", "Perform force restart; check analytics for kernel panic logs."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport AirDrop fails to detect nearby Apple devices after the recent update.", "Toggle AirDrop permissions to 'Everyone for 10 Minutes'; reset network settings."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Cellular data icon disappeared from control center on iOS 17.", "Check Carrier Settings Update in Settings > General > About."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Safari keeps crashing whenever I open a new tab on iPadOS.", "Clear history and website data in Settings > Safari."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Apple Music app won't open after updating my phone software.", "Restart iPhone; offload and reinstall the Apple Music application."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Face ID stopped working and says 'Not Available' after update.", "Force restart iPhone; inspect true depth camera for obstruction."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Screen brightness stays at minimum level and auto-brightness is broken post patch.", "Toggle Auto-Brightness off/on under Accessibility > Display & Text Size."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Storage space shows 'Other System Data' consuming 40GB after update.", "Backup device to iCloud/Mac and restore via Recovery Mode."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Haptic keyboard feedback stopped responding after system update.", "Ensure Haptic is enabled under Settings > Sounds & Haptics > Keyboard Feedback."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport CarPlay disconnects every time screen locks post iOS 17 update.", "Forget car connection in CarPlay settings and pair again via USB/Bluetooth."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Control Center layout reset to default after updating.", "Reconfigure Control Center items in Settings > Control Center."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Voicemail notification badge won't clear after listening to messages.", "Reset network settings or contact carrier to refresh voicemail box."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Battery health percentage dropped 5% overnight after software patch.", "Battery health recalibrates post update; monitor over 48 hours."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Personal Hotspot connection drops every 2 minutes after update.", "Toggle 'Maximize Compatibility' in Personal Hotspot settings."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Screen orientation auto-rotate is stuck in portrait mode.", "Check Portrait Orientation Lock in Control Center; restart device."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport StandBy mode feature not activating when phone is charging sideways.", "Verify StandBy switch under Settings > StandBy; use MagSafe charger."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport iMessage reactions and stickers failing to send after patch.", "Sign out of iMessage in Settings > Messages and sign back in."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Phone call audio sounds muffled through internal earpiece post update.", "Inspect earpiece mesh for debris; turn off Phone Noise Cancellation."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Mail app not fetching new emails automatically in background.", "Check Fetch New Data settings in Settings > Mail > Accounts."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Lock Screen widgets showing blank boxes after updating.", "Remove and re-add widgets on Lock Screen editing menu."),
    ("software_update", "AUTO-HANDLE", "@AppleSupport Update verification failed error appears every time I tap Install.", "Delete software update file from iPhone Storage and redownload."),

    # Hardware & Battery (30 unique examples)
    ("hardware_battery", "ESCALATE", "@AppleSupport My iPhone 13 screen went pitch black after dropping it on concrete.", "Inspect physical screen damage; direct to Genius Bar repair appointment."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Battery drops from 80% to 15% in less than 30 minutes of mild usage.", "Check Settings > Battery > Battery Health; send DM if capacity < 80%."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Phone gets extremely hot to touch while charging on official Apple MagSafe charger.", "Remove heavy case; check charging temperature; disconnect if prompt appears."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport My iPhone won't turn on at all after remaining plugged in overnight.", "Perform force restart (Volume Up, Volume Down, hold Side button)."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Phone back glass shattered into pieces after falling out of my pocket.", "Direct to apple.com/support/repair for out-of-warranty replacement options."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Battery percentage gets stuck at 80% while charging overnight.", "Optimized Battery Charging holds charge at 80% until needed; check settings."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Lightning charging port feels loose and cable drops out constantly.", "Inspect port for lint gently or visit Apple Store for physical port repair."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport iPhone screen has a permanent dark yellow tint near top left corner.", "Turn off True Tone/Night Shift; if tint persists, hardware inspection needed."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Device swelling from back cover near battery area. Is this safe?", "Stop using device immediately; disconnect charger; bring to Apple Store."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Camera lens glass cracked after dropping phone on rocks.", "Schedule repair appointment at nearest Apple Authorized Service Provider."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Phone vibrates continuously when plugged into wall charger.", "Check charging brick/cable compatibility; clean charging port."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Volume up physical button is stuck inside frame and won't click.", "Requires physical button mechanism repair at Genius Bar."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Battery drain is very high while using GPS navigation in car.", "GPS navigation uses heavy processing; use car charger while navigating."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Submerged phone in ocean water and speaker output sounds distorted.", "Dry device completely; do not charge while wet; seek hardware service."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Battery maximum capacity reads 74% and says Service recommended.", "Battery is degraded below standard 80%; recommend battery replacement."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Display panel shows pink lines running vertically down screen after drop.", "Internal display panel damage; book Genius Bar screen replacement."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Wireless charging pad not detecting my iPhone through standard case.", "Ensure case thickness is under 3mm and contains no metal components."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Power button snapped off side of device chassis.", "Physical housing damage requiring hardware repair appointment."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Device shuts down unexpectedly whenever battery drops to 20%.", "Battery peak performance capability compromised; check battery health."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Screen lifting away from aluminum frame near volume buttons.", "Potential battery expansion; stop charging and visit Genius Bar."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Flashlight LED won't turn on and icon is grayed out.", "Restart iPhone; verify device temperature is not too high."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Dropped phone in sink and touch screen no longer responds to touch.", "Liquid damage triage; schedule in-person diagnostic test."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Battery drains fast only when using cellular 5G network.", "Switch cellular voice & data option to 5G Auto or LTE in Settings."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Rear camera optical image stabilization buzzing loudly when opening app.", "Hardware camera OIS module fault requiring module repair."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Phone display flickers briefly when turning screen lock on/off.", "Adjust True Tone settings; restart phone."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Cracked front glass glass cuts fingers when swiping.", "Schedule screen replacement appointment immediately."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Charging cable gets very hot to touch near Lightning connector.", "Replace damaged charging cable immediately with MFi certified accessory."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Battery percentage drops 10% instantly upon unplugging from charger.", "Calibrate battery gauge by full discharge and unbroken 100% recharge."),
    ("hardware_battery", "ESCALATE", "@AppleSupport Speaker grill emits smoke or burning smell during fast charging.", "Unplug immediately; disconnect power; contact Apple Support safety team."),
    ("hardware_battery", "AUTO-HANDLE", "@AppleSupport Screen flickers yellow when brightness is turned below 20%.", "Turn off Night Shift; test display in standard lighting conditions."),

    # Account & iCloud (30 unique examples)
    ("account_icloud", "ESCALATE", "@AppleSupport Forgot my Apple ID password and locked out of primary account.", "Direct to iforgot.apple.com; escalate due to account security verification."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport iCloud storage shows full 50GB occupied even after deleting photos.", "Empty 'Recently Deleted' album in Photos; check iCloud Backup sizes."),
    ("account_icloud", "ESCALATE", "@AppleSupport I am not receiving the 2FA verification codes sent to my trusted phone number.", "Check carrier SMS blocking; direct to account recovery flow at iforgot.apple.com."),
    ("account_icloud", "ESCALATE", "@AppleSupport Someone logged into my Apple ID from another country! Account compromised!", "Change password immediately; revoke unknown trusted devices at appleid.apple.com."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport How do I upgrade my iCloud storage plan from 50GB to 200GB?", "Go to Settings > [Your Name] > iCloud > Manage Account Storage > Change Storage Plan."),
    ("account_icloud", "ESCALATE", "@AppleSupport Account Recovery request says it will take 3 days. Can you speed it up?", "Account recovery automated security wait period cannot be manually bypassed."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport Photos stopped syncing to iCloud across my Mac and iPad.", "Verify iCloud Photos is toggled ON under Settings > [Your Name] > iCloud > Photos."),
    ("account_icloud", "ESCALATE", "@AppleSupport Locked out of Apple ID due to typing incorrect password too many times.", "Reset password at iforgot.apple.com to unlock account access."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport How do I share my 2TB iCloud+ storage plan with my Family Sharing group?", "Go to Settings > Family > Subscriptions > iCloud+ > Share with Family."),
    ("account_icloud", "ESCALATE", "@AppleSupport Cannot sign out of iCloud because Sign Out button is grayed out.", "Disable Screen Time restriction under Content & Privacy Restrictions."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport How do I turn off iCloud Keychain password auto-fill on my phone?", "Toggle Keychain off under Settings > [Your Name] > iCloud > Passwords and Keychain."),
    ("account_icloud", "ESCALATE", "@AppleSupport My trusted device phone number was changed by unauthorized hacker.", "Escalate to Apple ID Fraud & Security Escalation team."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport Can I merge two separate Apple ID accounts into one unified account?", "Apple ID accounts cannot be merged; transfer data manually or use Family Sharing."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport iCloud Mail address receiving heavy spam emails. How to filter?", "Set up email rules in icloud.com mail web console."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport My iCloud Backup failed because of insufficient network connection.", "Connect to stable Wi-Fi network and plug iPhone into power source."),
    ("account_icloud", "ESCALATE", "@AppleSupport Forgot security questions for legacy Apple ID account created in 2012.", "Direct to iforgot.apple.com or verify identity with security team."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport How do I download a complete copy of all my data stored in iCloud?", "Visit privacy.apple.com and select 'Request a copy of your data'."),
    ("account_icloud", "ESCALATE", "@AppleSupport Activation Lock screen appears on refurbished phone bought second-hand.", "Requires original purchase receipt or seller to remove device from Find My."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport iCloud Drive files not appearing in Files app on iOS.", "Ensure iCloud Drive is enabled in Settings > [Your Name] > iCloud."),
    ("account_icloud", "ESCALATE", "@AppleSupport Security key hardware token lost for Apple ID Advanced Data Protection.", "Without trusted security key or recovery key, account access cannot be restored."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport How do I change the primary email address associated with my Apple ID?", "Sign in to appleid.apple.com and edit 'Sign-In and Security' email."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport Contacts vanished from phone after toggling iCloud account off.", "Toggle Contacts switch back ON under Settings > [Your Name] > iCloud."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport How to setup Legacy Contact for Apple ID digital inheritance?", "Go to Settings > [Your Name] > Sign-In & Security > Legacy Contact."),
    ("account_icloud", "ESCALATE", "@AppleSupport Verification code sent to Apple Watch not displaying code digits.", "Check Bluetooth connectivity between iPhone and Watch; resend SMS code."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport iCloud Private Relay not working on public Wi-Fi network.", "Some public Wi-Fi networks block Private Relay; disable in network settings."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport Notes app stopped syncing entries between iPhone and MacBook.", "Check Default Account setting in Settings > Notes."),
    ("account_icloud", "ESCALATE", "@AppleSupport Apple ID account disabled for security reasons pop-up.", "Direct user to reset password at iforgot.apple.com."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport How do I create a custom email domain with my iCloud+ subscription?", "Set up custom domain at icloud.com/settings or via iCloud Mail settings."),
    ("account_icloud", "AUTO-HANDLE", "@AppleSupport Hide My Email feature creating random address not sending emails.", "Check forwarded address under Settings > [Your Name] > iCloud > Hide My Email."),
    ("account_icloud", "ESCALATE", "@AppleSupport Child account age change request failing on Family Sharing admin console.", "Contact Apple Support to verify legal birth certificate details for child account."),

    # Billing & Refunds (30 unique examples)
    ("billing_refund", "ESCALATE", "@AppleSupport I was double charged $14.99 for my monthly Apple Music subscription on my credit card!", "Direct to reportaproblem.apple.com; escalate to human billing specialist."),
    ("billing_refund", "ESCALATE", "@AppleSupport My child accidentally spent $50 on in-app purchases. I need an immediate refund.", "Submit refund request at reportaproblem.apple.com; enable Screen Time purchase limits."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How do I cancel an active auto-renewing app subscription on my iPhone?", "Go to Settings > [Your Name] > Subscriptions > Cancel Subscription."),
    ("billing_refund", "ESCALATE", "@AppleSupport Charged $99.99 for annual app auto-renewal without pre-notification.", "Request refund via reportaproblem.apple.com; check subscription terms."),
    ("billing_refund", "ESCALATE", "@AppleSupport App Store gift card code shows 'already redeemed' when entered.", "Verify purchase receipt; escalate to Apple Gift Card billing support."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How to change credit card billing details on iTunes account?", "Update payment method in Settings > [Your Name] > Payment & Shipping."),
    ("billing_refund", "ESCALATE", "@AppleSupport Payment method declined error message on App Store despite sufficient funds.", "Check billing address details or contact credit card issuing bank."),
    ("billing_refund", "ESCALATE", "@AppleSupport Subscribed to 7-day free trial but credit card was billed full price immediately.", "Submit billing claim at reportaproblem.apple.com for trial fee refund."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How do I view purchase history for App Store transactions from last year?", "View purchase history under Settings > [Your Name] > Media & Purchases > View Account."),
    ("billing_refund", "ESCALATE", "@AppleSupport Unknown charge from 'APPLE.COM/BILL' for $29.99 on bank statement.", "Search purchase history at reportaproblem.apple.com; escalate if unrecognized."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How to redeem an Apple Gift Card code on iPhone?", "Open App Store > tap account icon > tap Redeem Gift Card or Code."),
    ("billing_refund", "ESCALATE", "@AppleSupport Refund request for accidentally purchased app was denied. How do I appeal?", "Escalate to Senior Billing Supervisor for refund appeal review."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport Can I use PayPal as primary payment method for App Store purchases?", "PayPal can be added under Settings > [Your Name] > Payment & Shipping."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How do I stop Family Sharing members from charging my credit card?", "Turn off Purchase Sharing under Settings > Family > Purchase Sharing."),
    ("billing_refund", "ESCALATE", "@AppleSupport Account balance of $15 gift card credit disappeared from account.", "Contact Apple Support billing team to audit gift card ledger balance."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How do I get an official invoice PDF receipt for my MacBook purchase?", "Download tax invoice at apple.com/orderstatus or check order confirmation email."),
    ("billing_refund", "ESCALATE", "@AppleSupport Charged twice for iCloud 200GB storage upgrade this month.", "Submit billing dispute at reportaproblem.apple.com."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport Free 6 months Apple Music offer with AirPods purchase not showing in app.", "Redeem offer within 90 days of pairing new AirPods in Apple Music app."),
    ("billing_refund", "ESCALATE", "@AppleSupport Card charged for app subscription after I already cancelled it last week.", "Verify cancellation timestamp; submit refund claim at reportaproblem.apple.com."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How to remove expired credit card from Apple Pay Wallet app?", "Open Wallet app > select card > tap menu button > select Remove Card."),
    ("billing_refund", "ESCALATE", "@AppleSupport App developer promised refund but money has not returned to bank account.", "Refunds take 3-5 business days; check status at reportaproblem.apple.com."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How do I turn off automatic renewal for Apple TV+ subscription?", "Cancel subscription under Settings > [Your Name] > Subscriptions."),
    ("billing_refund", "ESCALATE", "@AppleSupport Credit card charged in foreign currency with extra transaction fee.", "Update billing region in Apple ID account settings."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport Can I share Apple One subscription with family members?", "Apple One Family or Premier plans can be shared with up to 5 family members."),
    ("billing_refund", "ESCALATE", "@AppleSupport In-app purchase item bought for $19.99 never unlocked in game.", "Report missing item purchase at reportaproblem.apple.com or contact developer."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How do I set up Apple Pay on my new iPhone 15?", "Open Wallet app > tap + button > follow screen instructions to add card."),
    ("billing_refund", "ESCALATE", "@AppleSupport Bank flagged Apple Store transaction as fraudulent and blocked card.", "Contact issuing bank to authorize charge and update payment method."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How to check remaining balance on Apple Gift Card?", "Check balance online at apple.com/go/gcb/us or in Wallet app."),
    ("billing_refund", "ESCALATE", "@AppleSupport Billed for subscription after deleting app from home screen.", "Deleting an app does not cancel subscription; direct to subscription settings."),
    ("billing_refund", "AUTO-HANDLE", "@AppleSupport How do I change Apple Pay default payment card?", "Go to Settings > Wallet & Apple Pay > Default Card under Transaction Defaults."),

    # Accessory & Connectivity (30 unique examples)
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Left AirPod Pro completely dead and silent while right AirPod works fine.", "Clean charging tail/case contacts; reset AirPods by holding case setup button."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport My Apple Watch Series 8 will not pair with my new iPhone 15.", "Unpair Watch from old phone; reset network settings; restart both devices."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Right AirPod micro-phone has extreme static sound during calls.", "Reset AirPods; clean mic grill at bottom of stem."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport MagSafe wallet animation stopped showing when attached.", "Toggle Find My MagSafe wallet settings; restart iPhone."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Pencil 2nd gen drops Bluetooth connection every few minutes.", "Unpair Pencil in Bluetooth settings, re-attach magnetically to side of iPad."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirPods case light flashing orange continuously and won't pair.", "Reset AirPods case by holding setup button for 15 seconds until light flashes amber then white."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Watch battery drains from 100% to zero in 5 hours.", "Check Watch battery health; unpair and re-pair Watch as new device."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Magic Keyboard trackpad gesture scrolling reversed on iPad.", "Adjust Natural Scrolling setting under Settings > General > Trackpad."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirPods Max ear cups squeaking with high pitched feedback noise.", "Clean optical sensors; restart AirPods Max by holding noise control and crown button."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirTag battery low warning showing after 14 months of use.", "Replace AirTag CR2032 coin cell battery."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Watch screen stuck on red exclamation mark icon.", "Double-press side button and follow recovery steps on paired iPhone."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Beats Studio Buds audio dropping in right ear outdoor.", "Reset Beats buds; update Beats firmware via Beats app or iOS settings."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Magic Mouse won't connect via Bluetooth to Mac mini.", "Connect Magic Mouse to Mac via Lightning cable to re-establish Bluetooth pairing."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirPods Pro Noise Cancellation mode feels weaker after firmware update.", "Ensure silicone ear tips fit snugly; run Ear Tip Fit Test in settings."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Pencil tip worn down to metal stem. Do I replace tip?", "Replace worn Apple Pencil tip with official Apple Pencil Replacement Tips."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport MagSafe Battery Pack charging iPhone slowly at 5W.", "Connect MagSafe Battery Pack to 20W+ power adapter to charge at 15W."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Watch Digital Crown dial feels sticky and hard to turn.", "Rinse Digital Crown under warm tap water for 10 seconds while turning."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirPods automatic device switching between iPad and Mac not working.", "Verify 'Connect to this Mac/iPad' setting is set to Automatically on both devices."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport HomePod mini not responding in Home app and shows 'Not Responding'.", "Unplug HomePod mini power adapter, wait 10 seconds, and plug back in."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Siri remote for Apple TV 4K not scrolling or responding.", "Reset Siri Remote by pressing TV and Volume Down buttons simultaneously for 5 seconds."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirPods battery status popup not appearing on iPhone screen.", "Ensure Bluetooth is enabled; open AirPods case lid close to unlocked iPhone."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Watch heart rate monitor LED lights turned off.", "Verify Heart Rate privacy setting under Settings > Privacy > Health > Heart Rate."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Smart Keyboard Folio keys failing to type on iPad Pro.", "Clean Smart Connector contact dots on back of iPad with micro-fiber cloth."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirTag Precision Finding arrow not showing on iPhone 11.", "Precision Finding requires Ultra Wideband chip; ensure Location Services are ON."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirPods audio volume very low in left ear compared to right ear.", "Check Audio Balance slider under Settings > Accessibility > Audio/Visual."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Watch workout app GPS route tracking inaccurate.", "Calibrate Apple Watch by completing 20-minute outdoor walk or run."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport MagSafe Duo Charger folding hinge tearing near seam.", "Inspect physical cable/hinge wear; visit Apple Store for accessory replacement."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport AirPods mic picking up heavy wind noise during phone calls.", "Set Microphone selection setting to Always Left or Always Right ear."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Apple Pencil 1st gen won't charge when plugged into iPad port.", "Use Lightning adapter connector to charge Pencil directly from wall cable."),
    ("accessory_connectivity", "AUTO-HANDLE", "@AppleSupport Beats Fit Pro wingtips tearing after 6 months of active workouts.", "Replace wingtip accessories or visit Beats service channel."),

    # Repair & Store / Genius Bar (25 unique examples)
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport How can I book an in-person Genius Bar appointment at Fifth Avenue Apple Store?", "Book via support.apple.com or Apple Support iOS app under Store Appointments."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport My repair status for dispatch ID R987234 has not updated in 5 business days.", "Escalate to repair logistics team; check status at mysupport.apple.com."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport How do I cancel my Genius Bar appointment scheduled for tomorrow?", "Manage or cancel appointments via Apple Support app or confirmation email."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Is trade-in available at Apple Store without original product box?", "Yes, original box is not required for trade-in evaluation."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport What documents do I need to bring for an in-store Genius Bar appointment?", "Bring valid government photo ID, device, and back up data prior to appointment."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport Shipped phone for mail-in screen repair 2 weeks ago but tracking says pending delivery.", "Escalate to Apple Repair Mail-in Logistics team to track shipment."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Can I send someone else to pick up my repaired MacBook at Apple Store?", "Authorized pick-up person must be added to repair order and bring photo ID."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport How long does an in-store iPhone screen replacement usually take?", "Same-day screen repairs typically take 1 to 2 hours depending on store store load."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport Apple Store technician lost my original SIM tray during battery repair.", "Escalate to Store Service Manager for replacement SIM tray dispatch."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport How do I find an Apple Authorized Service Provider near me?", "Search authorized service locations at locate.apple.com."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Do I need to turn off Find My before submitting phone for repair?", "Yes, Find My must be turned OFF before technician service can begin."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport Received repaired phone from depot but back camera is broken now.", "Escalate to Repair Quality Assurance team for re-repair inspection."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport How do I check if my device is eligible for Self Service Repair program?", "Check manual and genuine parts eligibility at selfservicerepair.apple.com."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Can I reschedule my Genius Bar reservation to next weekend?", "Reschedule appointment time via Apple Support app or support website."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport Quote for MacBook logic board repair is higher than original estimate.", "Escalate to Repair Service Representative to review cost estimate breakdown."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport What is the cost of battery replacement for iPhone 12 out of warranty?", "Check estimated repair pricing at apple.com/support/repair/cost."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Can I drop off my iPhone for repair without an advance reservation?", "Walk-ins accepted based on availability; advance appointment strongly recommended."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport Service center holds my iPad for 3 weeks without spare part ETA.", "Escalate to Senior Repair Support Supervisor for parts expedited delivery."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport How do I prepare my Mac before bringing it into Genius Bar?", "Back up data via Time Machine; turn off FileVault password lock."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Does Apple Store offer loaner phones during multi-day repairs?", "Loaner phones subject to store availability for eligible AppleCare+ repairs."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport Store closed early due to weather emergency during my repair pick-up time.", "Escalate to Customer Support to reschedule pick-up window."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Can I trade in a broken iPhone with cracked screen for store credit?", "Cracked devices evaluated for reduced trade-in value or free recycling."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport How do I check warranty coverage status using my serial number?", "Check coverage details at checkcoverage.apple.com."),
    ("repair_store_genius", "ESCALATE", "@AppleSupport Mail-in repair box was damaged during courier transit to depot.", "Initiate damaged transit claim with shipping carrier and repair depot."),
    ("repair_store_genius", "AUTO-HANDLE", "@AppleSupport Where can I buy official Apple screen protector installation service?", "Belkin screen protector installation service available at official Apple Retail Stores."),

    # General Inquiry (25 unique examples)
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Is the 2nd generation Apple Pencil compatible with the new iPad Air 5th Gen?", "Yes, Apple Pencil 2nd gen is fully compatible with iPad Air 5th gen."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What are the holiday opening hours for Apple Store in Regent Street London?", "Check local store hours online at apple.com/retail/regentstreet."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Does iPhone 15 support dual eSIM functionality?", "Yes, iPhone 15 models support Dual eSIM with multiple active profiles."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What is the warranty coverage duration for newly purchased MacBooks?", "Includes 1-year limited hardware warranty and 90 days complimentary support."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What is the water resistance IP rating for iPhone 15 Pro?", "IP68 rated under IEC standard 60529 (maximum depth of 6 meters up to 30 minutes)."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Does AppleCare+ cover international repairs when traveling abroad?", "AppleCare+ provides global repair coverage in countries where service is available."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Can I use 30W USB-C charger from MacBook Air to charge my iPhone?", "Yes, Apple USB-C power adapters safely fast-charge compatible iPhone models."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What is the difference between AppleCare+ and standard 1-year Warranty?", "AppleCare+ adds accidental damage protection and 24/7 priority technical support."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport How many external monitors can I connect to MacBook Pro with M3 chip?", "M3 MacBook Pro supports one external display up to 6K resolution at 60Hz."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Is MagSafe charging compatible with iPhone 11 model?", "iPhone 11 supports Qi wireless charging; MagSafe magnetic alignment starts with iPhone 12."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Does iPad 10th generation support Stage Manager multitasking?", "Stage Manager is supported on iPad models with M1 chip or later and iPad Air 5th gen."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What audio codecs do AirPods Pro 2 support for lossless listening?", "AirPods Pro 2 support AAC codec and low-latency Lossless Audio with Vision Pro."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Can I transfer physical SIM card to eSIM on iPhone 15?", "Yes, convert physical SIM to eSIM via Settings > Cellular > Convert to eSIM."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What is the maximum storage capacity available for iPhone 15 Pro Max?", "Available in 256GB, 512GB, and 1TB storage configurations."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Does Apple offer educational student discount pricing on MacBooks?", "Yes, verified students and educators receive discount pricing at apple.com/us-hed/shop."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport How to enable Action Button customization on iPhone 15 Pro?", "Configure Action Button under Settings > Action Button."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Is Apple Card credit card available for international residents outside US?", "Apple Card is currently available exclusively to eligible United States residents."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Does Apple Fitness+ require an Apple Watch to subscribe?", "Apple Fitness+ can be used with just an iPhone on iOS 16.1 or later."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What is the battery video playback battery life specification for Mac Studio?", "Mac Studio is a desktop computer powered via direct AC outlet plug."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Can I use Apple Pay in web browser on Windows PC?", "Apple Pay on web requires Safari browser on iPhone, iPad, or Mac."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What cable connector is included in box with iPhone 15?", "Includes a 1-meter USB-C Charge Cable in the box."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Does Apple Vision Pro support prescription lens inserts?", "ZEISS Optical Inserts available for prescription vision correction."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport What is the weight difference between iPhone 14 Pro and iPhone 15 Pro?", "iPhone 15 Pro is ~19 grams lighter due to grade 5 titanium frame."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport Can I use AirTag to track my pet dog?", "AirTag is designed for tracking items and personal property."),
    ("general_inquiry", "AUTO-HANDLE", "@AppleSupport How do I check trade-in estimated value for my old iPhone 12?", "Check trade-in estimate value online at apple.com/shop/trade-in.")
]

def create_golden_set():
    print("=" * 60)
    print("      GOLDEN EVALUATION SET CREATION WORKFLOW")
    print("=" * 60)
    
    records = []
    for idx, (intent, action, msg, points) in enumerate(UNIQUE_200_CANDIDATES, 1):
        records.append({
            "id": f"gold_{idx:03d}",
            "message": msg,
            "true_intent": intent,
            "expected_action": action,
            "expected_reply_points": points,
            "notes": "candidate_for_human_verification - Curated AppleSupport domain example"
        })

    df = pd.DataFrame(records)
    
    # Ensure zero duplicates
    df = df.drop_duplicates(subset=['message']).copy()
    
    os.makedirs(os.path.dirname(GOLDEN_SET_PATH), exist_ok=True)
    df.to_csv(GOLDEN_SET_PATH, index=False)
    
    print(f"Golden evaluation candidate dataset saved to: {GOLDEN_SET_PATH}")
    print(f"Total Unique Examples: {len(df)}")
    print("\nIntent Distribution in Candidate Set:")
    intent_counts = df['true_intent'].value_counts()
    for intent, count in intent_counts.items():
        print(f"  - {intent:<25}: {count} examples ({count/len(df)*100:.1f}%)")
        
    print("\nAction Distribution:")
    action_counts = df['expected_action'].value_counts()
    for action, count in action_counts.items():
        print(f"  - {action:<15}: {count} examples ({count/len(df)*100:.1f}%)")
    print("=" * 60)
    return df

if __name__ == "__main__":
    create_golden_set()
