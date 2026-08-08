import qrcode

url = "https://github.com/Bombowy/PuzzleMind"

qr = qrcode.make(url)
qr.save("puzzlemind_github_qr.png")
