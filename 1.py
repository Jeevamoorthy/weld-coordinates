from ultralytics import SAM

model = SAM("mobile_sam.pt")

model.predict(
    source="C:\\Users\\jeeva\\OneDrive\\Attachments\\Desktop\\coordinates\\sample\\1.jpeg",
    show=True,
    device="cpu",
    save=True
)
