# URL 검증 테스트!

from onvif import ONVIFCamera

cam = ONVIFCamera('192.168.0.38', 8899, 'admin', 'Mbc320!!')

media = cam.create_media_service()
profiles = media.GetProfiles()

print("ALL PROFILES:")
for p in profiles:
    print("-", p.token, getattr(p, "Name", ""))

print("\nALL STREAM URIS:")
for p in profiles:
    stream = media.GetStreamUri({
        'StreamSetup': {
            'Stream': 'RTP-Unicast',
            'Transport': {'Protocol': 'RTSP'}
        },
        'ProfileToken': p.token
    })

    uri = stream.Uri
    print("\nPROFILE:", p.token, getattr(p, "Name", ""))
    print("RAW URI:", uri)
    print("REPR URI:", repr(uri))

    auth_uri = uri.replace("rtsp://", "rtsp://admin:Mbc320!!@")
    print("AUTH URI:", auth_uri)