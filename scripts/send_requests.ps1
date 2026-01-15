$url = "http://localhost:8000/predict"
for ($i = 1; $i -le 20; $i++) {
  $body = @{
    user_id = $i
    carat = 0.3 + (Get-Random -Minimum 0.0 -Maximum 0.2)
    cut = "Ideal"
    color = "E"
    clarity = "SI2"
    depth = 61.5
    table = 55
    x = 4.3
    y = 4.35
    z = 2.7
    true_price = 500 + (Get-Random -Minimum -50 -Maximum 50)
  } | ConvertTo-Json

  Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"
  Start-Sleep -Milliseconds 100
}
