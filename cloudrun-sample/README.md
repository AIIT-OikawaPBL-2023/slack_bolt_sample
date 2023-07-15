# ローカルで立ち上げる場合
## 前準備

### ngrok
ngrokでサーバーを立ち上げて `https://xxxx-xxxx-xxx-xxx-xxx.ngrok-free.app -> http://localhost:11111` となるようにする

```
ngrok http 11111
```

↓ngrokで生成されたURLをSlackアプリのevent-subscriptionsに登録する必要がある
https://api.slack.com/apps/A053QS5RJ1G/event-subscriptions?

※このときチャレンジ認証が行われる。下記のコードのように一回challengeを返してあげないと登録できない
```
    if 'challenge' in payload:
        # これはチャレンジリクエストです。アプリを確認するためにチャレンジ値で応答してください。
        # ref: https://qiita.com/masa_masa_ra/items/618779e698921cb53cec
        return payload['challenge']
```


## ローカルで立ち上げる場合

source .venv/bin/activate.fish
python main.py

## dockerで立ち上げる場合
docker build -t my_app -f Dockerfile_dev .
docker run -p 11111:11111 --env-file .env  my_app

## docker-composeで立ち上げる場合

### 事前準備

1. ngrok.yml.sampleをもとにngrok.ymlを作成する
1. ngrok(https://dashboard.ngrok.com/get-started/your-authtoken)にログインしてYour Authtokenをコピーしてngrok.ymlのauthtokenに設定する

### 起動方法

1. カレントディレクトリがcloudrun-sampleなことを確認して `docker compose up` で起動する
1. slackのevent-subscriptionsにngrokで生成されたURLを登録する必要があるので、ngrokが発行したURLが確認できるページ(http://localhost:4040/)をブラウザで開く
1. SlackのEvent Subscriptionに{表示されているURL}/slack/eventsを登録する
1. 以降の流れは通常のngrok使用時と変わらない
1. 起動したサービスを終了する際はcontrol + cで終了できる


## CloudRun

### for Linux or Mac
```bash
export SLACK_APP_TOKEN=xxxxxxxxxxxx
export SLACK_BOT_TOKEN=xxxxxxxxxxxx
export SLACK_SIGNING_SECRET=xxxxxxxxxxxx
gcloud builds submit --tag gcr.io/$PROJECT_ID/helloworld
gcloud run deploy helloworld --image gcr.io/$PROJECT_ID/helloworld --platform managed --update-env-vars SLACK_BOT_TOKEN=$SLACK_BOT_TOKEN,SLACK_SIGNING_SECRET=$SLACK_SIGNING_SECRET
```

### for Windows
```powershell
set SLACK_APP_TOKEN=xxxxxxxxxxxx
set SLACK_BOT_TOKEN=xxxxxxxxxxxx
set SLACK_SIGNING_SECRET=xxxxxxxxxxxx
gcloud builds submit --tag gcr.io/%PROJECT_ID%/helloworld
gcloud run deploy helloworld --image gcr.io/%PROJECT_ID%/helloworld --platform managed --update-env-vars SLACK_BOT_TOKEN=%SLACK_BOT_TOKEN%,SLACK_SIGNING_SECRET=%SLACK_SIGNING_SECRET%
```

## 設定
https://{ngrokやcloudrunのエンドポイント}/slack/events