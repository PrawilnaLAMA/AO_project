import torch
import torch.cuda as cuda
import traceback

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    
    # Test 1: Podstawowe operacje na CUDA
    print("\n=== Test 1: Basic CUDA operations ===")
    try:
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x + y
        print("✓ Basic tensor operations work")
        print(f"Tensor on: {z.device}")
    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        traceback.print_exc()
    
    # Test 2: Prosty model
    print("\n=== Test 2: Simple model ===")
    try:
        simple_model = torch.nn.Conv2d(3, 64, 3, 1, 1).cuda()
        dummy_input = torch.randn(1, 3, 160, 160).cuda()
        with torch.no_grad():
            output = simple_model(dummy_input)
        print("✓ Simple model works")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Simple model failed: {e}")
        traceback.print_exc()
    
    # Test 3: Facenet model
    print("\n=== Test 3: Facenet model ===")
    try:
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().cuda()
        print("✓ Facenet model loaded")
        
        # Test inference
        with torch.no_grad():
            output = model(dummy_input)
        print("✓ Facenet inference works")
        print(f"Embedding shape: {output.shape}")
    except Exception as e:
        print(f"✗ Facenet model failed: {e}")
        traceback.print_exc()

else:
    print("CUDA not available")