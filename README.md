# Sign2flow – A dynamic sign language translator
Status: *Experimental*. This project is still in progress. Right now, it works on short videos and can turn dynamic signing into phrases. But it doesn’t yet know when someone is not signing, and sometimes it mistakes faces for hands. It’s improving quickly, so expect regular updates.

## What this is
sign2flow takes short clips or live streams of sign language and translates them into natural phrases. Instead of just looking at still handshapes, it focuses on the movement, flow, and timing of signs—because in real sign language, meaning often comes from motion.

## What it does

-Looks at a video and highlights the movements of hands and arms.

-Understands the context of how signs unfold over time and turns them into phrases, not just single words.

-Cleans up the output a little so you see the most likely phrase, along with how confident the system is.

## How it works

-Seeing the motion: Frames from the video are sampled, and the system looks closely at where hands and arms are moving. It pays special attention to when signs start, hold, and change.

-Understanding the flow: A temporal model connects those frames together, so the system understands the bigger picture instead of treating each frame in isolation.

-Producing a phrase: The model’s step-by-step predictions are smoothed out into a readable phrase, so the result feels clearer and less jumpy.

## What it doesn’t do yet

-Idle moments: It doesn’t yet recognize when no signing is happening, so it might guess phrases even when the person is still.

-Hands vs. faces: Sometimes it mistakes a face for a hand, especially in tricky or crowded videos.

-Full conversations: It works best for short clips and phrases, not long sentences or flowing conversations.

## Why this matters
Sign language is about movement, not just static hand poses. By focusing on motion and timing, sign2flow gets closer to how signing actually works in real life—fluid, rhythmic, and dynamic. This project is a first step toward more natural and useful translation tools that can grow with better detection and more diverse training.

## Road ahead

-Teach the system to recognize “no-sign” states so it stays quiet when nothing is being signed.

-Improve hand detection so faces don’t get mistaken for hands.

-Smooth out predictions further so quick mistakes don’t make the output unstable.

-Train and test on more varied signing styles, backgrounds, and lighting for stronger performance in real-world use.
