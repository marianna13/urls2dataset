import os
import uuid
import requests
import io
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse import detect_encoding
import hashlib
from urllib.parse import urljoin, urlparse


_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "max-age=0",
    "referer": "https://www.google.comt",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
}
_HEADERS = {}


def get_extension(url: str) -> str:
    """Parse the URL using the urlparse method
    Get the file name and extension from the parsed URL
    Return the file extension"""
    parsed_url = urlparse(url)
    try:
        filename, file_ext = parsed_url.path.rsplit(".", maxsplit=1)
    except ValueError:
        file_ext = ""
    return file_ext


def parser_bytes(url, tree):
    """
    Some notes: csrc,chash are the current source and hash of the image/video/iframe
    """
    iframedict, vids, imgs, auds = dict(), dict(), dict(), dict()
    page_config = {"img_count": 0, "vid_count": 0, "aud_count": 0, "iframe_count": 0}

    for ele in tree.body.get_elements_by_tag_name("nav"):
        ele.parent.remove_child(ele)

    for ele in tree.body.get_elements_by_tag_name("img"):
        csrc = urljoin(url, ele.getattr("src"))
        chash = str(hashlib.md5((csrc).encode()).hexdigest())

        imgs[f"###img#{page_config['img_count']}###"] = (chash, get_extension(csrc), csrc)
        ele.setattr("alt", f"###img#{page_config['img_count']}###")
        page_config["img_count"] += 1

    for ele in tree.body.get_elements_by_tag_name("iframe"):
        csrc = urljoin(url, ele.getattr("src"))
        chash = str(hashlib.md5((csrc).encode()).hexdigest())

        iframedict[f"###iframe#{page_config['iframe_count']}###"] = (chash, get_extension(csrc), csrc)
        nele = tree.create_element("img")
        nele["src"] = csrc
        nele.setattr("alt", f"###iframe#{page_config['iframe_count']}###")
        page_config["iframe_count"] += 1
        ele.parent.append_child(nele)
        ele.parent.replace_child(nele, ele)

    for ele in tree.body.get_elements_by_tag_name("video"):

        if len(ele.get_elements_by_tag_name("source")) > 0:
            mele = ele.get_elements_by_tag_name("source")
            csrc = mele[0].getattr("src")
            csrc = urljoin(url, csrc)
            chash = str(hashlib.md5((csrc).encode()).hexdigest())

            vids[f"###video#{page_config['vid_count']}###"] = (chash, get_extension(csrc), csrc)
            nele = tree.create_element("img")
            nele["src"] = csrc
            nele.setattr("alt", f"###video#{page_config['vid_count']}###")
            page_config["vid_count"] += 1
            ele.parent.insert_before(nele, ele)
            ele.parent.remove_child(ele)

        if ele.getattr("src"):
            csrc = ele.getattr("src")
            csrc = urljoin(url, csrc)
            chash = str(hashlib.md5((csrc).encode()).hexdigest())
            vids[f"###video#{page_config['vid_count']}###"] = (chash, get_extension(csrc), csrc)
            nele = tree.create_element("img")
            nele.setattr("src", csrc)
            nele.setattr("alt", f"###video#{page_config['vid_count']}###")
            page_config["vid_count"] += 1
            ele.parent.append_child(nele)
            ele.parent.replace_child(nele, ele)

    for ele in tree.body.get_elements_by_tag_name("audio"):

        if len(ele.get_elements_by_tag_name("source")) > 0:
            mele = ele.get_elements_by_tag_name("source")

            csrc = mele[0].getattr("src")
            csrc = urljoin(url, csrc)
            chash = str(hashlib.md5((csrc).encode()).hexdigest())

            auds[f"###audio#{page_config['aud_count']}###"] = (chash, get_extension(csrc), csrc)
            nele = tree.create_element("img")
            nele.setattr("src", csrc)
            nele.setattr("alt", f"###audio#{page_config['aud_count']}###")
            page_config["aud_count"] += 1
            ele.parent.insert_before(nele, ele)
            ele.parent.remove_child(ele)

        if ele.getattr("src"):

            csrc = ele.getattr("src")
            csrc = urljoin(url, csrc)
            chash = str(hashlib.md5((csrc).encode()).hexdigest())

            auds[f"###audio#{page_config['aud_count']}###"] = (chash, get_extension(csrc), csrc)
            nele = tree.create_element("img")
            nele["src"] = csrc
            nele.setattr("alt", f"###audio#{page_config['aud_count']}###")
            ele.parent.append_child(nele)
            ele.parent.replace_child(nele, ele)
            page_config["aud_count"] += 1

    return tree, {
        # 'page_config': page_config,
        "imgs": imgs,
        "vids": vids,
        "auds": auds,
        "iframedict": iframedict,
    }


class CCDownloader:
    def __init__(self, config):
        self.config = config

    def __call__(self, html_bytes):
        text = None
        try:
            if self.config.get("media_elems"):
                encoding = detect_encoding(html_bytes)
                tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
                tree, media = parser_bytes(url, tree)
            if self.config.get("save_media_struct"):
                text = extract_plain_text(
                    tree,
                    preserve_formatting=False,
                    main_content=False,
                    list_bullets=False,
                    alt_texts=True,
                    links=False,
                    form_fields=False,
                    noscript=False,
                )
            else:
                text = extract_plain_text(html_bytes.decode())
            error = None

        except Exception as err:
            # print(err)
            error = str(err)


class URLDownloader:
    def __init__(self, timeout, headers=None, config={}):
        self.timeout = timeout
        self.headers = headers if headers is not None else _HEADERS
        self.config = config

    def __call__(self, url):

        media = {}

        text = None
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            error = f"response {resp.status_code}"
            if resp.status_code == 200:
                html_bytes = resp.content
                if self.config.get("media_elems"):
                    encoding = detect_encoding(html_bytes)
                    tree = HTMLTree.parse_from_bytes(html_bytes, encoding)
                    tree, media = parser_bytes(url, tree)
                if self.config.get("save_media_struct"):
                    text = extract_plain_text(
                        tree,
                        preserve_formatting=False,
                        main_content=False,
                        list_bullets=False,
                        alt_texts=True,
                        links=False,
                        form_fields=False,
                        noscript=False,
                    )
                else:
                    text = extract_plain_text(html_bytes.decode())
                error = None

        except Exception as err:
            # print(err)
            error = str(err)

        return text, media, error


class DataReader:
    """URLs data reader provide data for a URL"""

    def __init__(self, dl_timeout, tmp_dir, config, common_crawl=False) -> None:
        if common_crawl:
            self.downloader = CCDownloader(config)
        else:
            self.downloader = URLDownloader(dl_timeout, config=config)

    def __call__(self, row):
        key, url = row

        text, media, error_message = self.downloader(url)

        return key, text, media, error_message
