import mediapipe as mp

print("mediapipe path:", mp.__file__)
print("has solutions:", hasattr(mp, "solutions"))
print(dir(mp))
