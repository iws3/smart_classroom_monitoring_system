# Section 1: Getting the Phone's Video to the Laptop - Checkpoint Answers

I have gone through the first part of the document, successfully ran the classroom feed from my phone (IP: `192.168.1.115:8080/video`), and implemented the ResilientStream class for auto-reconnection. Here are my answers to the questions:

## Question 1: Why do the phone and laptop need to be on the same Wi-Fi network?

**Answer:** The phone acts as a local HTTP server, and its IP address is only accessible within that same local network subnet.

## Question 2: What line of code from Docs 1-2 is the only thing that changes?

**Answer:** Only `cap = cv2.VideoCapture(0)` changes to `cap = cv2.VideoCapture(PHONE_STREAM_URL)` - all MediaPipe detection code remains identical.

## Question 3: Why is a "retry on failure" pattern especially important?

**Answer:** Wi-Fi drops are common during a class period, and auto-reconnection ensures the system runs unattended without crashing or requiring manual restart.

✅ **Section 1 Complete** - Ready to proceed to Section 2: Multi-Person Tracking
