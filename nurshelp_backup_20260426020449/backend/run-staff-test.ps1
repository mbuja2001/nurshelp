Write-Host "TEST STAFF LIST ENDPOINT FOR ST MERCY HOSPITAL" -ForegroundColor Cyan
Write-Host ("=" * 80)

# Step 1: Login
Write-Host "`nSTEP 1: Login as St Mercy Hospital Admin" -ForegroundColor Yellow
Write-Host ("-" * 80)
Write-Host "Email: alice.johnson@stmercy.hospital"
Write-Host "Password: password123"
Write-Host ""

$loginBody = @{
    email = "alice.johnson@stmercy.hospital"
    password = "password123"
} | ConvertTo-Json

try {
    $loginRes = Invoke-WebRequest -Uri "http://localhost:5001/api/auth/login" `
        -Method POST `
        -Headers @{"Content-Type" = "application/json"} `
        -Body $loginBody `
        -UseBasicParsing
    
    $loginData = $loginRes.Content | ConvertFrom-Json
    
    Write-Host "Login Response:" -ForegroundColor Green
    Write-Host "  Status: $($loginRes.StatusCode)"
    Write-Host "  User: $($loginData.user.name)"
    Write-Host "  Hospital: $($loginData.user.hospitalName)"
    Write-Host "  Role: $($loginData.user.role)"
    Write-Host "  Hospital ID: $($loginData.user.hospitalId)"
    
    if (-not $loginData.token) {
        Write-Host "`nLOGIN FAILED!" -ForegroundColor Red
        exit 1
    }
    
    $token = $loginData.token
    Write-Host "  Token: $($token.Substring(0, 30))..." -ForegroundColor Green
    
    # Step 2: Get staff list
    Write-Host "`nSTEP 2: Query Staff List Endpoint" -ForegroundColor Yellow
    Write-Host ("-" * 80)
    Write-Host "URL: GET /api/manager/staff/list"
    Write-Host ""
    
    $staffRes = Invoke-WebRequest -Uri "http://localhost:5001/api/manager/staff/list" `
        -Method GET `
        -Headers @{"Authorization" = "Bearer $token"} `
        -UseBasicParsing
    
    $staffData = $staffRes.Content | ConvertFrom-Json
    
    Write-Host "Staff List Response:" -ForegroundColor Green
    Write-Host "  Status: $($staffRes.StatusCode)"
    Write-Host "  Success: $($staffData.success)"
    Write-Host "  Count: $($staffData.count)" -ForegroundColor $(if ($staffData.count -gt 0) { "Green" } else { "Yellow" })
    Write-Host ""
    
    if (-not $staffData.success) {
        Write-Host "STAFF QUERY FAILED!" -ForegroundColor Red
        exit 1
    }
    
    if ($staffData.staff -and $staffData.staff.Count -gt 0) {
        Write-Host "STAFF MEMBERS AT ST MERCY HOSPITAL:" -ForegroundColor Green
        $idx = 0
        foreach ($staff in $staffData.staff) {
            $idx++
            Write-Host ""
            Write-Host "$idx. $($staff.firstName) $($staff.lastName)" -ForegroundColor Green
            Write-Host "   Email: $($staff.email)"
            Write-Host "   Role: $($staff.systemRole)"
            $dept = if ($staff.assignment.department) { $staff.assignment.department } elseif ($staff.department) { $staff.department } else { 'N/A' }
            Write-Host "   Department: $dept"
            $status = if ($staff.status) { $staff.status } else { 'N/A' }
            Write-Host "   Status: $status"
        }
    } else {
        Write-Host "NO STAFF MEMBERS FOUND!" -ForegroundColor Yellow
    }
    
    # Step 3: Test filters
    Write-Host "`nSTEP 3: Test Filters" -ForegroundColor Yellow
    Write-Host ("-" * 80)
    
    $filters = @(
        @{desc = "Filter by Role NURSE"; url = "/api/manager/staff/list?role=NURSE"},
        @{desc = "Filter by Role PHYSICIAN"; url = "/api/manager/staff/list?role=PHYSICIAN"},
        @{desc = "Search: Wong"; url = "/api/manager/staff/list?search=Wong"}
    )
    
    foreach ($filter in $filters) {
        $filterRes = Invoke-WebRequest -Uri "http://localhost:5001$($filter.url)" `
            -Method GET `
            -Headers @{"Authorization" = "Bearer $token"} `
            -UseBasicParsing
        
        $filterData = $filterRes.Content | ConvertFrom-Json
        
        Write-Host "`n$($filter.desc):" -ForegroundColor Cyan
        Write-Host "  Count: $($filterData.count)"
        
        if ($filterData.staff -and $filterData.staff.Count -gt 0) {
            foreach ($s in $filterData.staff) {
                Write-Host "    - $($s.firstName) $($s.lastName) ($($s.systemRole))"
            }
        } else {
            Write-Host "    (No results)"
        }
    }
    
    Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
    Write-Host "TEST COMPLETE`n" -ForegroundColor Green
    
} catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.InnerException) {
        Write-Host "Detail: $($_.Exception.InnerException.Message)" -ForegroundColor Red
    }
    exit 1
}
