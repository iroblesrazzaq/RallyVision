# Tennis Point Detection - Feature Vector Documentation

This document provides a comprehensive breakdown of the 360-element feature vector used as input to the LSTM model for tennis point detection. The vector contains features for two players (near and far) with 180 features each.

## Feature Vector Structure Overview

- **Total Features**: 360 elements
- **Player 1 (Near)**: Elements 0-179
- **Player 2 (Far)**: Elements 180-359

## Player 1 (Near Player) Features (Elements 0-179)

### Presence and Basic Position Information
0: Player presence flag (1.0 = present, -1.0 = absent)
1: Bounding box x1 coordinate (top-left corner)
2: Bounding box y1 coordinate (top-left corner)
3: Bounding box x2 coordinate (bottom-right corner)
4: Bounding box y2 coordinate (bottom-right corner)
5: Centroid x coordinate (center of bounding box)
6: Centroid y coordinate (center of bounding box)

### Player Motion Features
7: Player velocity x component (vx)
8: Player velocity y component (vy)
9: Player acceleration x component (ax)
10: Player acceleration y component (ay)
11: Player speed (magnitude of velocity vector)
12: Player acceleration magnitude (magnitude of acceleration vector)

### Keypoint Position Data (Elements 13-46)
13: Keypoint 0 (nose) x coordinate
14: Keypoint 0 (nose) y coordinate
15: Keypoint 1 (left_eye) x coordinate
16: Keypoint 1 (left_eye) y coordinate
17: Keypoint 2 (right_eye) x coordinate
18: Keypoint 2 (right_eye) y coordinate
19: Keypoint 3 (left_ear) x coordinate
20: Keypoint 3 (left_ear) y coordinate
21: Keypoint 4 (right_ear) x coordinate
22: Keypoint 4 (right_ear) y coordinate
23: Keypoint 5 (left_shoulder) x coordinate
24: Keypoint 5 (left_shoulder) y coordinate
25: Keypoint 6 (right_shoulder) x coordinate
26: Keypoint 6 (right_shoulder) y coordinate
27: Keypoint 7 (left_elbow) x coordinate
28: Keypoint 7 (left_elbow) y coordinate
29: Keypoint 8 (right_elbow) x coordinate
30: Keypoint 8 (right_elbow) y coordinate
31: Keypoint 9 (left_wrist) x coordinate
32: Keypoint 9 (left_wrist) y coordinate
33: Keypoint 10 (right_wrist) x coordinate
34: Keypoint 10 (right_wrist) y coordinate
35: Keypoint 11 (left_hip) x coordinate
36: Keypoint 11 (left_hip) y coordinate
37: Keypoint 12 (right_hip) x coordinate
38: Keypoint 12 (right_hip) y coordinate
39: Keypoint 13 (left_knee) x coordinate
40: Keypoint 13 (left_knee) y coordinate
41: Keypoint 14 (right_knee) x coordinate
42: Keypoint 14 (right_knee) y coordinate
43: Keypoint 15 (left_ankle) x coordinate
44: Keypoint 15 (left_ankle) y coordinate
45: Keypoint 16 (right_ankle) x coordinate
46: Keypoint 16 (right_ankle) y coordinate

### Keypoint Confidence Data (Elements 47-63)
47: Keypoint 0 (nose) confidence
48: Keypoint 1 (left_eye) confidence
49: Keypoint 2 (right_eye) confidence
50: Keypoint 3 (left_ear) confidence
51: Keypoint 4 (right_ear) confidence
52: Keypoint 5 (left_shoulder) confidence
53: Keypoint 6 (right_shoulder) confidence
54: Keypoint 7 (left_elbow) confidence
55: Keypoint 8 (right_elbow) confidence
56: Keypoint 9 (left_wrist) confidence
57: Keypoint 10 (right_wrist) confidence
58: Keypoint 11 (left_hip) confidence
59: Keypoint 12 (right_hip) confidence
60: Keypoint 13 (left_knee) confidence
61: Keypoint 14 (right_knee) confidence
62: Keypoint 15 (left_ankle) confidence
63: Keypoint 16 (right_ankle) confidence

### Keypoint Velocity Data (Elements 64-97)
64: Keypoint 0 (nose) velocity x component
65: Keypoint 0 (nose) velocity y component
66: Keypoint 1 (left_eye) velocity x component
67: Keypoint 1 (left_eye) velocity y component
68: Keypoint 2 (right_eye) velocity x component
69: Keypoint 2 (right_eye) velocity y component
70: Keypoint 3 (left_ear) velocity x component
71: Keypoint 3 (left_ear) velocity y component
72: Keypoint 4 (right_ear) velocity x component
73: Keypoint 4 (right_ear) velocity y component
74: Keypoint 5 (left_shoulder) velocity x component
75: Keypoint 5 (left_shoulder) velocity y component
76: Keypoint 6 (right_shoulder) velocity x component
77: Keypoint 6 (right_shoulder) velocity y component
78: Keypoint 7 (left_elbow) velocity x component
79: Keypoint 7 (left_elbow) velocity y component
80: Keypoint 8 (right_elbow) velocity x component
81: Keypoint 8 (right_elbow) velocity y component
82: Keypoint 9 (left_wrist) velocity x component
83: Keypoint 9 (left_wrist) velocity y component
84: Keypoint 10 (right_wrist) velocity x component
85: Keypoint 10 (right_wrist) velocity y component
86: Keypoint 11 (left_hip) velocity x component
87: Keypoint 11 (left_hip) velocity y component
88: Keypoint 12 (right_hip) velocity x component
89: Keypoint 12 (right_hip) velocity y component
90: Keypoint 13 (left_knee) velocity x component
91: Keypoint 13 (left_knee) velocity y component
92: Keypoint 14 (right_knee) velocity x component
93: Keypoint 14 (right_knee) velocity y component
94: Keypoint 15 (left_ankle) velocity x component
95: Keypoint 15 (left_ankle) velocity y component
96: Keypoint 16 (right_ankle) velocity x component
97: Keypoint 16 (right_ankle) velocity y component

### Keypoint Acceleration Data (Elements 98-131)
98: Keypoint 0 (nose) acceleration x component
99: Keypoint 0 (nose) acceleration y component
100: Keypoint 1 (left_eye) acceleration x component
101: Keypoint 1 (left_eye) acceleration y component
102: Keypoint 2 (right_eye) acceleration x component
103: Keypoint 2 (right_eye) acceleration y component
104: Keypoint 3 (left_ear) acceleration x component
105: Keypoint 3 (left_ear) acceleration y component
106: Keypoint 4 (right_ear) acceleration x component
107: Keypoint 4 (right_ear) acceleration y component
108: Keypoint 5 (left_shoulder) acceleration x component
109: Keypoint 5 (left_shoulder) acceleration y component
110: Keypoint 6 (right_shoulder) acceleration x component
111: Keypoint 6 (right_shoulder) acceleration y component
112: Keypoint 7 (left_elbow) acceleration x component
113: Keypoint 7 (left_elbow) acceleration y component
114: Keypoint 8 (right_elbow) acceleration x component
115: Keypoint 8 (right_elbow) acceleration y component
116: Keypoint 9 (left_wrist) acceleration x component
117: Keypoint 9 (left_wrist) acceleration y component
118: Keypoint 10 (right_wrist) acceleration x component
119: Keypoint 10 (right_wrist) acceleration y component
120: Keypoint 11 (left_hip) acceleration x component
121: Keypoint 11 (left_hip) acceleration y component
122: Keypoint 12 (right_hip) acceleration x component
123: Keypoint 12 (right_hip) acceleration y component
124: Keypoint 13 (left_knee) acceleration x component
125: Keypoint 13 (left_knee) acceleration y component
126: Keypoint 14 (right_knee) acceleration x component
127: Keypoint 14 (right_knee) acceleration y component
128: Keypoint 15 (left_ankle) acceleration x component
129: Keypoint 15 (left_ankle) acceleration y component
130: Keypoint 16 (right_ankle) acceleration x component
131: Keypoint 16 (right_ankle) acceleration y component

### Keypoint Motion Magnitude Data (Elements 132-165)
132: Keypoint 0 (nose) speed (velocity magnitude)
133: Keypoint 1 (left_eye) speed (velocity magnitude)
134: Keypoint 2 (right_eye) speed (velocity magnitude)
135: Keypoint 3 (left_ear) speed (velocity magnitude)
136: Keypoint 4 (right_ear) speed (velocity magnitude)
137: Keypoint 5 (left_shoulder) speed (velocity magnitude)
138: Keypoint 6 (right_shoulder) speed (velocity magnitude)
139: Keypoint 7 (left_elbow) speed (velocity magnitude)
140: Keypoint 8 (right_elbow) speed (velocity magnitude)
141: Keypoint 9 (left_wrist) speed (velocity magnitude)
142: Keypoint 10 (right_wrist) speed (velocity magnitude)
143: Keypoint 11 (left_hip) speed (velocity magnitude)
144: Keypoint 12 (right_hip) speed (velocity magnitude)
145: Keypoint 13 (left_knee) speed (velocity magnitude)
146: Keypoint 14 (right_knee) speed (velocity magnitude)
147: Keypoint 15 (left_ankle) speed (velocity magnitude)
148: Keypoint 16 (right_ankle) speed (velocity magnitude)
149: Keypoint 0 (nose) acceleration magnitude
150: Keypoint 1 (left_eye) acceleration magnitude
151: Keypoint 2 (right_eye) acceleration magnitude
152: Keypoint 3 (left_ear) acceleration magnitude
153: Keypoint 4 (right_ear) acceleration magnitude
154: Keypoint 5 (left_shoulder) acceleration magnitude
155: Keypoint 6 (right_shoulder) acceleration magnitude
156: Keypoint 7 (left_elbow) acceleration magnitude
157: Keypoint 8 (right_elbow) acceleration magnitude
158: Keypoint 9 (left_wrist) acceleration magnitude
159: Keypoint 10 (right_wrist) acceleration magnitude
160: Keypoint 11 (left_hip) acceleration magnitude
161: Keypoint 12 (right_hip) acceleration magnitude
162: Keypoint 13 (left_knee) acceleration magnitude
163: Keypoint 14 (right_knee) acceleration magnitude
164: Keypoint 15 (left_ankle) acceleration magnitude
165: Keypoint 16 (right_ankle) acceleration magnitude

### Anatomical Limb Lengths (Elements 166-179)
166: Left shoulder to left elbow limb length
167: Left elbow to left wrist limb length
168: Right shoulder to right elbow limb length
169: Right elbow to right wrist limb length
170: Left hip to left knee limb length
171: Left knee to left ankle limb length
172: Right hip to right knee limb length
173: Right knee to right ankle limb length
174: Left shoulder to right shoulder limb length
175: Left hip to right hip limb length
176: Left shoulder to left hip limb length
177: Right shoulder to right hip limb length
178: Right shoulder to left shoulder limb length
179: Right hip to left hip limb length

## Player 2 (Far Player) Features (Elements 180-359)

### Presence and Basic Position Information
180: Player presence flag (1.0 = present, -1.0 = absent)
181: Bounding box x1 coordinate (top-left corner)
182: Bounding box y1 coordinate (top-left corner)
183: Bounding box x2 coordinate (bottom-right corner)
184: Bounding box y2 coordinate (bottom-right corner)
185: Centroid x coordinate (center of bounding box)
186: Centroid y coordinate (center of bounding box)

### Player Motion Features
187: Player velocity x component (vx)
188: Player velocity y component (vy)
189: Player acceleration x component (ax)
190: Player acceleration y component (ay)
191: Player speed (magnitude of velocity vector)
192: Player acceleration magnitude (magnitude of acceleration vector)

### Keypoint Position Data (Elements 193-226)
193: Keypoint 0 (nose) x coordinate
194: Keypoint 0 (nose) y coordinate
195: Keypoint 1 (left_eye) x coordinate
196: Keypoint 1 (left_eye) y coordinate
197: Keypoint 2 (right_eye) x coordinate
198: Keypoint 2 (right_eye) y coordinate
199: Keypoint 3 (left_ear) x coordinate
200: Keypoint 3 (left_ear) y coordinate
201: Keypoint 4 (right_ear) x coordinate
202: Keypoint 4 (right_ear) y coordinate
203: Keypoint 5 (left_shoulder) x coordinate
204: Keypoint 5 (left_shoulder) y coordinate
205: Keypoint 6 (right_shoulder) x coordinate
206: Keypoint 6 (right_shoulder) y coordinate
207: Keypoint 7 (left_elbow) x coordinate
208: Keypoint 7 (left_elbow) y coordinate
209: Keypoint 8 (right_elbow) x coordinate
210: Keypoint 8 (right_elbow) y coordinate
211: Keypoint 9 (left_wrist) x coordinate
212: Keypoint 9 (left_wrist) y coordinate
213: Keypoint 10 (right_wrist) x coordinate
214: Keypoint 10 (right_wrist) y coordinate
215: Keypoint 11 (left_hip) x coordinate
216: Keypoint 11 (left_hip) y coordinate
217: Keypoint 12 (right_hip) x coordinate
218: Keypoint 12 (right_hip) y coordinate
219: Keypoint 13 (left_knee) x coordinate
220: Keypoint 13 (left_knee) y coordinate
221: Keypoint 14 (right_knee) x coordinate
222: Keypoint 14 (right_knee) y coordinate
223: Keypoint 15 (left_ankle) x coordinate
224: Keypoint 15 (left_ankle) y coordinate
225: Keypoint 16 (right_ankle) x coordinate
226: Keypoint 16 (right_ankle) y coordinate

### Keypoint Confidence Data (Elements 227-243)
227: Keypoint 0 (nose) confidence
228: Keypoint 1 (left_eye) confidence
229: Keypoint 2 (right_eye) confidence
230: Keypoint 3 (left_ear) confidence
231: Keypoint 4 (right_ear) confidence
232: Keypoint 5 (left_shoulder) confidence
233: Keypoint 6 (right_shoulder) confidence
234: Keypoint 7 (left_elbow) confidence
235: Keypoint 8 (right_elbow) confidence
236: Keypoint 9 (left_wrist) confidence
237: Keypoint 10 (right_wrist) confidence
238: Keypoint 11 (left_hip) confidence
239: Keypoint 12 (right_hip) confidence
240: Keypoint 13 (left_knee) confidence
241: Keypoint 14 (right_knee) confidence
242: Keypoint 15 (left_ankle) confidence
243: Keypoint 16 (right_ankle) confidence

### Keypoint Velocity Data (Elements 244-277)
244: Keypoint 0 (nose) velocity x component
245: Keypoint 0 (nose) velocity y component
246: Keypoint 1 (left_eye) velocity x component
247: Keypoint 1 (left_eye) velocity y component
248: Keypoint 2 (right_eye) velocity x component
249: Keypoint 2 (right_eye) velocity y component
250: Keypoint 3 (left_ear) velocity x component
251: Keypoint 3 (left_ear) velocity y component
252: Keypoint 4 (right_ear) velocity x component
253: Keypoint 4 (right_ear) velocity y component
254: Keypoint 5 (left_shoulder) velocity x component
255: Keypoint 5 (left_shoulder) velocity y component
256: Keypoint 6 (right_shoulder) velocity x component
257: Keypoint 6 (right_shoulder) velocity y component
258: Keypoint 7 (left_elbow) velocity x component
259: Keypoint 7 (left_elbow) velocity y component
260: Keypoint 8 (right_elbow) velocity x component
261: Keypoint 8 (right_elbow) velocity y component
262: Keypoint 9 (left_wrist) velocity x component
263: Keypoint 9 (left_wrist) velocity y component
264: Keypoint 10 (right_wrist) velocity x component
265: Keypoint 10 (right_wrist) velocity y component
266: Keypoint 11 (left_hip) velocity x component
267: Keypoint 11 (left_hip) velocity y component
268: Keypoint 12 (right_hip) velocity x component
269: Keypoint 12 (right_hip) velocity y component
270: Keypoint 13 (left_knee) velocity x component
271: Keypoint 13 (left_knee) velocity y component
272: Keypoint 14 (right_knee) velocity x component
273: Keypoint 14 (right_knee) velocity y component
274: Keypoint 15 (left_ankle) velocity x component
275: Keypoint 15 (left_ankle) velocity y component
276: Keypoint 16 (right_ankle) velocity x component
277: Keypoint 16 (right_ankle) velocity y component

### Keypoint Acceleration Data (Elements 278-311)
278: Keypoint 0 (nose) acceleration x component
279: Keypoint 0 (nose) acceleration y component
280: Keypoint 1 (left_eye) acceleration x component
281: Keypoint 1 (left_eye) acceleration y component
282: Keypoint 2 (right_eye) acceleration x component
283: Keypoint 2 (right_eye) acceleration y component
284: Keypoint 3 (left_ear) acceleration x component
285: Keypoint 3 (left_ear) acceleration y component
286: Keypoint 4 (right_ear) acceleration x component
287: Keypoint 4 (right_ear) acceleration y component
288: Keypoint 5 (left_shoulder) acceleration x component
289: Keypoint 5 (left_shoulder) acceleration y component
290: Keypoint 6 (right_shoulder) acceleration x component
291: Keypoint 6 (right_shoulder) acceleration y component
292: Keypoint 7 (left_elbow) acceleration x component
293: Keypoint 7 (left_elbow) acceleration y component
294: Keypoint 8 (right_elbow) acceleration x component
295: Keypoint 8 (right_elbow) acceleration y component
296: Keypoint 9 (left_wrist) acceleration x component
297: Keypoint 9 (left_wrist) acceleration y component
298: Keypoint 10 (right_wrist) acceleration x component
299: Keypoint 10 (right_wrist) acceleration y component
300: Keypoint 11 (left_hip) acceleration x component
301: Keypoint 11 (left_hip) acceleration y component
302: Keypoint 12 (right_hip) acceleration x component
303: Keypoint 12 (right_hip) acceleration y component
304: Keypoint 13 (left_knee) acceleration x component
305: Keypoint 13 (left_knee) acceleration y component
306: Keypoint 14 (right_knee) acceleration x component
307: Keypoint 14 (right_knee) acceleration y component
308: Keypoint 15 (left_ankle) acceleration x component
309: Keypoint 15 (left_ankle) acceleration y component
310: Keypoint 16 (right_ankle) acceleration x component
311: Keypoint 16 (right_ankle) acceleration y component

### Keypoint Motion Magnitude Data (Elements 312-345)
312: Keypoint 0 (nose) speed (velocity magnitude)
313: Keypoint 1 (left_eye) speed (velocity magnitude)
314: Keypoint 2 (right_eye) speed (velocity magnitude)
315: Keypoint 3 (left_ear) speed (velocity magnitude)
316: Keypoint 4 (right_ear) speed (velocity magnitude)
317: Keypoint 5 (left_shoulder) speed (velocity magnitude)
318: Keypoint 6 (right_shoulder) speed (velocity magnitude)
319: Keypoint 7 (left_elbow) speed (velocity magnitude)
320: Keypoint 8 (right_elbow) speed (velocity magnitude)
321: Keypoint 9 (left_wrist) speed (velocity magnitude)
322: Keypoint 10 (right_wrist) speed (velocity magnitude)
323: Keypoint 11 (left_hip) speed (velocity magnitude)
324: Keypoint 12 (right_hip) speed (velocity magnitude)
325: Keypoint 13 (left_knee) speed (velocity magnitude)
326: Keypoint 14 (right_knee) speed (velocity magnitude)
327: Keypoint 15 (left_ankle) speed (velocity magnitude)
328: Keypoint 16 (right_ankle) speed (velocity magnitude)
329: Keypoint 0 (nose) acceleration magnitude
330: Keypoint 1 (left_eye) acceleration magnitude
331: Keypoint 2 (right_eye) acceleration magnitude
332: Keypoint 3 (left_ear) acceleration magnitude
333: Keypoint 4 (right_ear) acceleration magnitude
334: Keypoint 5 (left_shoulder) acceleration magnitude
335: Keypoint 6 (right_shoulder) acceleration magnitude
336: Keypoint 7 (left_elbow) acceleration magnitude
337: Keypoint 8 (right_elbow) acceleration magnitude
338: Keypoint 9 (left_wrist) acceleration magnitude
339: Keypoint 10 (right_wrist) acceleration magnitude
340: Keypoint 11 (left_hip) acceleration magnitude
341: Keypoint 12 (right_hip) acceleration magnitude
342: Keypoint 13 (left_knee) acceleration magnitude
343: Keypoint 14 (right_knee) acceleration magnitude
344: Keypoint 15 (left_ankle) acceleration magnitude
345: Keypoint 16 (right_ankle) acceleration magnitude

### Anatomical Limb Lengths (Elements 346-359)
346: Left shoulder to left elbow limb length
347: Left elbow to left wrist limb length
348: Right shoulder to right elbow limb length
349: Right elbow to right wrist limb length
350: Left hip to left knee limb length
351: Left knee to left ankle limb length
352: Right hip to right knee limb length
353: Right knee to right ankle limb length
354: Left shoulder to right shoulder limb length
355: Left hip to right hip limb length
356: Left shoulder to left hip limb length
357: Right shoulder to right hip limb length
358: Right shoulder to left shoulder limb length
359: Right hip to left hip limb length

## Missing Data Representation

- **Missing players**: All features set to -1.0 (except presence flag which is -1.0)
- **Missing keypoints**: Position (-1, -1), confidence -1.0, velocity (0, 0), acceleration (0, 0), speed -1.0, acceleration magnitude -1.0
- **Unavailable motion data**: Velocity and acceleration components set to 0.0, magnitudes set to -1.0 or 0.0 as appropriate

## Feature Categories Summary

1. **Presence Information**: 1 feature per player
2. **Bounding Box Data**: 4 features per player
3. **Centroid Position**: 2 features per player
4. **Player Motion**: 4 vector components + 2 magnitudes per player
5. **Keypoint Positions**: 34 features per player (17 keypoints × 2 coordinates)
6. **Keypoint Confidences**: 17 features per player
7. **Keypoint Velocities**: 34 features per player (17 keypoints × 2 components)
8. **Keypoint Accelerations**: 34 features per player (17 keypoints × 2 components)
9. **Keypoint Motion Magnitudes**: 34 features per player (17 speeds + 17 acceleration magnitudes)
10. **Anatomical Limb Lengths**: 14 features per player

**Total**: 180 features per player × 2 players = 360 features