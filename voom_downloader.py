import sys
import os
import time
from urllib.parse import urlparse

import requests
from playwright.sync_api import sync_playwright

if len(sys.argv) < 2:
    print("用法: python voom_downloader.py <LINE VOOM 文章網址>")
    sys.exit(1)

url = sys.argv[1]

SAVE_DIR = "voom_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(img_url, idx):
    parsed = urlparse(img_url)
    ext = os.path.splitext(parsed.path)[1] or ".jpg"
    r = requests.get(img_url, timeout=30)
    r.raise_for_status()
    with open(os.path.join(SAVE_DIR, f"{idx}{ext}"), "wb") as f:
        f.write(r.content)

def pick_largest_image(page, candidates):
    best = None
    best_area = 0
    for img in candidates:
        try:
            box = img.bounding_box()
            if not box:
                continue
            area = box["width"] * box["height"]
            if area > best_area:
                best_area = area
                best = img
        except Exception:
            continue
    return best

def safe_click(element, label):
    if not element:
        return False
    try:
        element.scroll_into_view_if_needed(timeout=2000)
    except Exception:
        pass
    try:
        element.click(timeout=2000, force=True)
        return True
    except Exception as e:
        print(f"[warn] 點擊失敗 ({label}): {e}")
        return False

def get_active_viewer_info(page):
    # Prefer the active swiper slide image inside the viewer
    active_slide = page.query_selector(".vw_media_viewer .swiper-slide-active")
    if active_slide:
        idx = active_slide.get_attribute("data-swiper-slide-index")
        img = active_slide.query_selector(".vw_media_viewer_item img")
        if img:
            src = img.get_attribute("src") or img.get_attribute("data-src")
            return img, src, idx

    # Fallback to any viewer image
    viewer_imgs = page.query_selector_all(".vw_media_viewer img.media_image, .vw_media_viewer img[src*='line-scdn']")
    img = pick_largest_image(page, viewer_imgs)
    if not img:
        return None, None, None
    src = img.get_attribute("src") or img.get_attribute("data-src")
    return img, src, None

def get_viewer_unique_indices(page):
    try:
        indices = page.eval_on_selector_all(
            ".vw_media_viewer .swiper-slide[data-swiper-slide-index]",
            "els => Array.from(new Set(els.map(e => e.getAttribute('data-swiper-slide-index')))).filter(Boolean)"
        )
        return indices or []
    except Exception:
        return []

def find_next_button(page):
    # LINE VOOM viewer next button from DOM (inline viewer + modal viewer)
    selectors = [
        "button.button_content_next:not([disabled])",
        ".vw_media_viewer button.button_content_next:not([disabled])",
        "button.button_move.button_next:not([disabled])",
        "button[aria-label*='Next']:not([disabled])",
        "button[aria-label*='下一張']:not([disabled])",
        "button[aria-label*='下一則']:not([disabled])",
    ]
    for sel in selectors:
        btn = page.query_selector(sel)
        if btn:
            return btn

    # Inline viewer chevrons (often 2 buttons: prev/next)
    group_buttons = page.query_selector_all(".vw_move_button_group button:not([disabled])")
    if group_buttons:
        return group_buttons[-1]
    return None

def collect_slide_image_urls(page):
    # Grab all images from swiper slides without needing to open the viewer
    urls = []
    imgs = page.query_selector_all(
        ".media_layout .swiper-slide img.media_image, "
        ".media_layout .swiper-slide img[src*='line-scdn']"
    )
    for img in imgs:
        src = img.get_attribute("src")
        if src:
            urls.append(src)
    # Keep order but dedupe
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered

def get_active_slide_src(page):
    img = page.query_selector(".swiper-slide-active img.media_image, .swiper-slide-active img[src*='line-scdn']")
    if not img:
        return None
    return img.get_attribute("src")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    print("啟動瀏覽器並打開 LINE VOOM 文章...")
    page.goto(url)

    # 等候頁面圖片載入
    page.wait_for_selector("img", timeout=20000)

    downloaded = set()
    index = 1

    # 一律點開檢視器，逐張按下一頁
    # 點進第一張主圖：鎖定 viewer 內的 media_image（避免頭像 img.image）
    target = page.query_selector(
        ".vw_viewer_content_wrap .media_layout .swiper-slide-active "
        ".media_item.type_viewer img.media_image"
    )
    if not target:
        target = page.query_selector(
            ".vw_viewer_content_wrap .media_layout .swiper-slide-active "
            ".media_item.type_viewer"
        )
    if not target:
        target = page.query_selector(
            ".vw_viewer_content_wrap .media_layout img.media_image"
        )
    if not target:
        target = page.query_selector(".vw_viewer_content_wrap .media_layout")
    if not target:
        target = page.query_selector(".media_top_inner img.media_image")
    if not target:
        all_imgs = page.query_selector_all("article img, main img, img")
        target = pick_largest_image(page, all_imgs)
    if target:
        safe_click(target, "open_viewer_target")
    else:
        print("找不到可點擊的圖片，請確認網址是否為單一文章。")
        browser.close()
        sys.exit(1)

    # 等待檢視器出現
    try:
        page.wait_for_selector(".vw_media_viewer", timeout=10000)
    except Exception:
        print("尚未進入檢視器，嘗試繼續。")

    total_indices = get_viewer_unique_indices(page)
    if total_indices:
        print(f"偵測到 {len(total_indices)} 張（依 swiper indices）")
    seen_indices = set()

    while True:

        # 取目前檢視器中的主圖（避免縮圖/頭貼）
        img, src, idx = get_active_viewer_info(page)
        print(f"目前主圖: {src} | slide={idx}")
        if idx is not None:
            seen_indices.add(idx)
        if src and src not in downloaded:
            print(f"下載第 {index} 張...")
            download_image(src, index)
            downloaded.add(src)
            index += 1

        # 嘗試切換到下一張：以鍵盤右鍵為主
        prev_src = src
        prev_idx = idx
        print("按鍵盤右鍵切換")
        viewer = page.query_selector(".vw_media_viewer")
        if viewer:
            safe_click(viewer, "viewer")
        elif img:
            safe_click(img, "active_image")
        page.keyboard.press("ArrowRight")

        # 等待 0.5 秒後判斷是否切換成功
        time.sleep(0.5)
        _, new_src, new_idx = get_active_viewer_info(page)
        if (new_idx is not None and new_idx != prev_idx) or (new_src and new_src != prev_src):
            if new_src and new_src not in downloaded:
                print(f"下載第 {index} 張...")
                download_image(new_src, index)
                downloaded.add(new_src)
                index += 1
        else:
            if total_indices and len(seen_indices) >= len(total_indices):
                print("已走完所有 swiper slide。")
            else:
                print("已沒有更多圖片可下載。")
            break

    browser.close()

print("下載完成。")
