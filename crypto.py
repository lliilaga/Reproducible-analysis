
import zipfile
import numpy as np

# Step 1: Read the timing measurements
with zipfile.ZipFile("traces.npz") as z:
    raw = z.read("traces.npy")

records = np.frombuffer(
    raw[192:],
    dtype=[
        ("pos", "u1"),
        ("guess", "u1"),
        ("time", "<f4")
    ]
)

# Step 2: Recover one byte at a time
key = []

for pos in range(32):
    groups = [
        records[
            (records["pos"] == pos) &
            (records["guess"] == g)
        ]["time"]
        for g in range(256)
    ]

    scores = []

    for g, candidate in enumerate(groups):
        other = np.concatenate(groups[:g] + groups[g + 1:])

        delta = candidate.mean() - other.mean()

        se = np.sqrt(
            candidate.var(ddof=1) / len(candidate)
            + other.var(ddof=1) / len(other)
        )

        score = delta / se
        scores.append((score, g))

    score, guess = max(scores)
    key.append(guess)

    print(f"Position {pos}: guess={guess:#04x}, score={score:.3f}")

# Step 3: Display the recovered bytes
print("Recovered bytes:", bytes(key))
print("Hex:", bytes(key).hex())
