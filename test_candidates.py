
import requests

BASE_URL = "http://localhost:8000/predict"

candidates = [
    # Cats
    ("Cat 1", "https://storage.googleapis.com/kagglesdsdata/datasets/3829311/6633136/PetImages/Cat/10003.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260130%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260130T045756Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=bdecde88c586ac0c54b697513958c060e322f472c6d1d1aeeec0621f9d80c7925d6d7b39841c3c4cea9c8180a530e8fc4cd8a79cd0f0c657f6bedd4782268cb0e87a0716001bb8371ddc7bea2f2819c63a7db17a0bad711cb6392cb5b62c5c294a37ff36c5d904302f20fd08de37309df33fbb7b09d719ab063c2fd7bbac0514c040e994b4dd7ade30b66f914da1309ba316a14643be19a06c2d6f28ebc6d547f82eeacc27db3935521eea5488d9e64fbab8b36fca587a95f61f6852e8a78f878a74f7da66f253c0cdbed117e85c7f74cbb47c70ab7ca8022552a6da24c0f16b79810a9bb5c0ba33169ee395dac224a60565fff224e9117b2a1e3cab3f75e271"),
    ("Cat 2", "https://storage.googleapis.com/kagglesdsdata/datasets/3829311/6633136/PetImages/Cat/10001.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260130%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260130T045755Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=1565f4dbc7501b080ab8060be6e4b4e27b593885d7cb936b22676ef84222fde15f98f7818b1feb7521eeaff6ff90eb65ae5eefcd31b07d306740f18e56dd9391298801774b4cdf2d26bf58d6d08aace9eda6219103c554d96505faff2aacba467f3ee918c66c8dc068be32246f01766943aef9eae2ef96f0a2d06414c8cfa425538b688bd35f6d68b3fbfffe73e603da5db0c662b8d6d80ccf85232364bd7b6189336700ab69deab912893c4e8acfda41c65eaa919748b9d9326f0ab1b7f73af4351cec52ea48e11e38db29dca0f6f1963d270fb6bea6b8c0dc0374f42eba8ada7945ea712faa5170101763fb6d54982ffbf3de63fe96cf1f864f59e1444c3ed"),
    
    # Dogs
    ("Dog 1", "https://storage.googleapis.com/kagglesdsdata/datasets/3829311/6633136/PetImages/Dog/10005.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260131%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260131T164603Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=33ad3c7ccc83db46785238b1e58afd4a0e201f41e98cabe6a2034dcdcdb89365633ec2a2761d77e1c892ec0f00df9b4ff4f67a3c657ae2a38c46d1956da88955d87073722fb6271a745bcd669c8e3c733f2cdf046f024f68fc7357c0ec84b700d5c5e2ad615239c2b986601458684238b70854e12af814e3d043a9a334c80878d3d6c88a23dccb5b7f4dd06a23ae886b4c231b03041f15395511ba3ec4630bddd12447511ab68eacd0c74f6f1385aaeeb2acc13ab78b5b4834d1f53e6b9b089549680db2b1dd7cfdfe438beb55ff74f3ac93ef2bcb6cf66e27e446b8291bf4bc28f7528dd7b2c0b73281ae89ecd13bed98ca4ebfa216422416c76d911bc03b60"),
    ("Dog 2", "https://storage.googleapis.com/kagglesdsdata/datasets/3829311/6633136/PetImages/Dog/10003.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260131%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260131T164603Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=5e99c58457dd127be9533159cf2057d84376a9a07b9343f7fe33748a36b3974fdba52b72bd9c15479f761c1b92c34116805632df4eb33cbf683fac7ec22f2a5982230a7889358b0e953a2362daa8765535a4953dbd16270e88738e0bf6497bb412bceb73046f45107bda5db6c53825811113966f13246a67213b082086acf7e73c917ad1bab47235534c0b8c04aaba04ee21befa26beeee3d8944e703688bed94587daf27d9352f45389c29dc829bac54b0c76a481d22fec38a63c6aae23acfb6b3bdeed3a37f169350909c214eef25376fa908ab2b7016995ca0fb7d40f87231c9c7570c2652e4a942485940f5cb9d889e45c893c280eace5907ca11889f835"),

    # Neutral URLs (Unsplash)
    ("Unsplash Cat 1", "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"),
    ("Unsplash Dog 1", "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"),
]

for name, url in candidates:
    try:
        resp = requests.post(BASE_URL, json={"image_url": url})
        if resp.status_code == 200:
            print(f"{name}: {resp.json()}")
        else:
            print(f"{name}: Failed {resp.status_code}")
    except Exception as e:
        print(f"{name}: Error {e}")
