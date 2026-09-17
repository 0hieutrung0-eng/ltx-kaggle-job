name: Trigger Kaggle Render

on:
  repository_dispatch:
    types: [run-kaggle]

jobs:
  run-kaggle:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install Kaggle CLI
        run: pip install kaggle

      - name: Configure Kaggle Credentials
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          mkdir -p ~/.kaggle
          cat > ~/.kaggle/kaggle.json << EOF
          {
            "username": "$KAGGLE_USERNAME",
            "key": "$KAGGLE_KEY"
          }
          EOF
          chmod 600 ~/.kaggle/kaggle.json

      - name: Inject Google Service Account vào script (Base64)
        env:
          GOOGLE_SERVICE_ACCOUNT: ${{ secrets.GOOGLE_SERVICE_ACCOUNT }}
        run: |
          python << 'EOF'
          import os
          import json
          import base64

          secret = os.environ["GOOGLE_SERVICE_ACCOUNT"]
          
          # Kiểm tra JSON hợp lệ
          json.loads(secret)

          # Encode sang Base64
          b64 = base64.b64encode(secret.encode("utf-8")).decode("utf-8")

          script_path = "kaggle_script/render_script.py"

          with open(script_path, "r", encoding="utf-8") as f:
              content = f.read()

          placeholder = "___GOOGLE_SERVICE_ACCOUNT_B64_PLACEHOLDER___"
          if placeholder not in content:
              raise ValueError("Không tìm thấy placeholder Base64 trong render_script.py")

          new_content = content.replace(placeholder, b64)

          with open(script_path, "w", encoding="utf-8") as f:
              f.write(new_content)

          print("✅ Đã inject Service Account (Base64) vào render_script.py thành công")
          EOF

      - name: Generate Dynamic Slug
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
        run: |
          TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
          DYNAMIC_SLUG="ltx-render-$TIMESTAMP"
          
          echo "--> Generating new notebook ID: $KAGGLE_USERNAME/$DYNAMIC_SLUG"
          
          python -c "
          import json, os
          username = os.environ['KAGGLE_USERNAME']
          timestamp = '$TIMESTAMP'
          slug = '$DYNAMIC_SLUG'
          
          with open('kernel-metadata.json', 'r') as f:
              data = json.load(f)
              
          data['id'] = f'{username}/{slug}'
          data['title'] = f'LTX Render {timestamp}'
          
          with open('kernel-metadata.json', 'w') as f:
              json.dump(data, f, indent=2)
          "

      - name: Push New Code to Kaggle
        env:
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
        run: |
          kaggle kernels push -p .
